from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..models.plumbing import PlumbingItem, PageData, ExtractionResult
import pdfplumber
from unstructured.partition.pdf import partition_pdf
import os
import json
import gc
import torch
import psutil
import time
import multiprocessing
import pandas as pd

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def measure_resource_usage(func, *args, **kwargs):
    """Measure memory and CPU usage of a function"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    return {
        "memory_delta": end_memory - start_memory,
        "cpu_usage": psutil.cpu_percent(),
        "time_taken": end_time - start_time,
        "result": result
    }

def process_element(args):
    """Process a single element in parallel"""
    element, target_page = args
    if hasattr(element, "page_number") and element.page_number == target_page:
        return str(element)
    return None

def process_elements_parallel(elements, target_page):
    """Process elements in parallel with limited processes"""
    # Use half the available CPU cores to prevent system overload
    num_processes = max(1, multiprocessing.cpu_count() // 2)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_element, [(e, target_page) for e in elements])
    return [r for r in results if r is not None]

def process_pdf_chunk(args):
    """Process a chunk of the PDF in parallel"""
    pdf_path, start_page, end_page = args
    return partition_pdf(pdf_path, include_page_breaks=True, start_page=start_page, end_page=end_page)

def process_pdf_parallel(pdf_path: str, target_page: int) -> List[Any]:
    """Process PDF in parallel chunks"""
    # Get total pages
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    # Create chunks of 2 pages each
    chunk_size = 2
    chunks = [(pdf_path, i, min(i + chunk_size - 1, total_pages)) 
              for i in range(1, total_pages + 1, chunk_size)]
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_pdf_chunk, chunks)
    
    # Flatten results and filter for target page
    elements = []
    for chunk_elements in results:
        for element in chunk_elements:
            if hasattr(element, "page_number") and element.page_number == target_page:
                elements.append(element)
    return elements

class PDFExtractionChain:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        self.memory = ConversationBufferMemory()
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=PlumbingItem)
        
        # Create extraction prompt
        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract plumbing information from this text. Follow these exact rules:

1. Input Format Handling:
   - If input uses ":" or "+" as separators, convert to pipe format
   - If input has multiple lines per item, combine into single line
   - If input has headers (e.g., "HHWS:", "Pipe type:"), extract relevant information
   - Remove any bullet points (*) or other markers
   - Remove any indentation or extra spaces
   - Remove any trailing "mounting_type" text
   - Handle "BE=" notation as mounting type
   - Handle "HUH" and "BC" prefixes in model numbers
   - Handle cases where model number and dimensions are mixed (e.g., "1 1/2"ø HHWR")
   - Convert "N/A" or "(no quantity)" to "1" for quantity field
   - Separate model numbers from dimensions (e.g., "1 1/2"ø" should be split into model number and dimensions)
   - Handle "OM-" and "BC" model number prefixes
   - Handle "HH" and "CH" mounting types

2. Output Format:
   - Each item must be on a single line
   - Fields must be in this exact order: type|quantity|model_number|dimensions|mounting_type
   - Fields must be separated by a single pipe character (|)
   - No extra spaces around separators
   - No headers or section numbers
   - No empty lines between items
   - Example: pipe|1|HHWS|1 1/2 inch|19 ft - 3 3/8 inch

3. Field Rules:
   a) Type (first field):
      - Must be one of: pipe, valve, fitting, fixture, accessory
      - If input starts with model number (e.g., "HHWS:"), extract type from context
      - For items with "ø" symbol, use "pipe" as type
      - Convert "Piping:" to "pipe"
      - Convert "Valve Package:" to "valve"
      - If type is missing, use "pipe" as default

   b) Quantity (second field):
      - Must be a number (default to "1" if not specified)
      - Remove any special characters or units
      - Convert to string format
      - Convert "N/A" to "1"
      - Convert "(no quantity)" to "1"
      - Convert "quantity: N/A" to "1"
      - Convert "-" to "1"
      - Convert "2 1/2" to "2.5"
      - If quantity is missing or empty, use "1"
      - If quantity is not a valid number, use "1"

   c) Model Number (third field):
      - Must be one of: HHWS, HHWR, CWS, CWR, CHWS, CHWR, CD, M7, OM-*, BC *, HUH *
      - Extract from type field if present (e.g., "HHWS:" → "HHWS")
      - Remove any quotes or brackets
      - Remove any special characters
      - Keep "HUH", "BC", and "OM-" prefixes in model numbers
      - For mixed format (e.g., "1 1/2"ø HHWR"), extract only the model number (HHWR)
      - Remove any dimension information from model number (e.g., "1 1/2"ø" should not be in model number)
      - If model number is missing, use blank ("")

   d) Dimensions (fourth field):
      - Format: "X inch" or "X ft - Y inch"
      - Extract from size field (e.g., "1 1/2"ø" → "1 1/2 inch")
      - For mixed format (e.g., "1 1/2"ø HHWR"), extract only the dimension (1 1/2 inch)
      - Remove "ø" symbol
      - Convert "inches" to "inch"
      - Convert " to "inch"
      - Convert ' to "ft"
      - Always use "ft -" (with space and hyphen)
      - Keep fractions as is
      - Always include "inch" suffix
      - If dimension is missing, use blank ("")
      - For multiple dimensions, use only the first one
      - Never include mounting type in dimensions

   e) Mounting Type (fifth field):
      - Format: "X ft - Y inch" or "HH" or "CH"
      - Extract from BE= field if present
      - Remove "BE=" prefix
      - Convert " to "inch"
      - Convert ' to "ft"
      - Always use "ft -" (with space and hyphen)
      - Keep fractions as is
      - Always include "inch" suffix
      - Keep "HH" and "CH" as mounting types
      - If no mounting type, use blank ("")
      - For multiple mounting types, use only the first one

4. Special Cases:
   - For items with multiple mounting types, use only the first one
   - For items with combined types (e.g., "HWS & HWR"), create separate entries
   - For items with model number in type field, extract to correct field
   - For items with dimensions in mounting field, move to correct field
   - For items with mounting in dimensions field, move to correct field
   - Remove any trailing "mounting_type" text
   - Keep "HUH", "BC", and "OM-" prefixes in model numbers
   - Ensure consistent spacing in fractions (e.g., "3 3/8" not "3 3/8")
   - For mixed format (e.g., "1 1/2"ø HHWR"), split into separate fields
   - Always convert "N/A" or "(no quantity)" to "1" for quantity field
   - Never include dimension information in model number field
   - Never include model number information in dimension field
   - For multiple dimensions, use only the first one
   - Keep "HH" and "CH" as mounting types

5. General Rules:
   - Remove any empty fields (replace with blank "")
   - Remove any extra commas or separators
   - Ensure consistent spacing around separators
   - Do not include headers or section numbers
   - Each item should be on a single line
   - No trailing separators
   - No empty lines between items
   - No duplicate items
   - Do not mix up field order
   - Ensure consistent unit formatting
   - Remove any trailing text after the last field
   - Always convert "N/A" or "(no quantity)" to "1" for quantity field
   - Keep model numbers and dimensions separate
   - Never mix model numbers with dimensions
   - For multiple dimensions, use only the first one
   - Keep "HH" and "CH" as mounting types
            
            Text: {text}
            
Return ONLY the standardized items in this exact format:
            type|quantity|model_number|dimensions|mounting_type
            
Do not include any explanations or additional text.
            """
        )
        
        # Create extraction chain using RunnableSequence
        self.extraction_chain = RunnableSequence(
            self.extraction_prompt | self.llm
        )

    def _process_text_with_llm(self, text: str) -> List[PlumbingItem]:
        """Process text with LLM in chunks based on complete items."""
        # First, try to identify tables or structured sections
        table_markers = ['Table', 'Section', 'Page', 'Item', 'Qty', 'Type', 'Model', 'Dim', 'Mount']
        has_tables = any(marker in text for marker in table_markers)
        
        if has_tables:
            # If we have tables, process them as a single chunk to preserve structure
            chunks = [text]
        else:
            # Split text into paragraphs or sections
            sections = [s.strip() for s in text.split('\n\n') if s.strip()]
            
            # Group sections into chunks of reasonable size
            chunks = []
            current_chunk = []
            current_size = 0
            
            for section in sections:
                section_size = len(section)
                # If adding this section would exceed 2000 characters
                # or if we have a complete table/section, start a new chunk
                if current_size + section_size > 2000 or any(marker in section for marker in table_markers):
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [section]
                    current_size = section_size
                else:
                    current_chunk.append(section)
                    current_size += section_size
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # If we have too few chunks, try a different splitting strategy
            if len(chunks) < 2:
                # Fall back to a more granular splitting
                chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        
        all_items = []
        for chunk in chunks:
            try:
                # Use the new RunnableSequence format
                response = self.extraction_chain.invoke({"text": chunk})
                items = self._parse_llm_response(response)
                all_items.extend(items)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        return all_items

    def extract_from_pdf(self, pdf_path: str, target_page: int = 5) -> ExtractionResult:
        """Extract data from PDF using both pdfplumber and unstructured.
        
        Args:
            pdf_path: Path to the PDF file
            target_page: Page number to process (1-based index). Defaults to 5.
        """
        pages_data = []
        
        print(f"\n=== Starting PDF Processing for Page {target_page} ===")
        
        # Step 1: Process with pdfplumber
        print("\n1. Processing with PDFPlumber...")
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if target_page > total_pages:
                raise ValueError(f"PDF only has {total_pages} pages, cannot process page {target_page}")
            
            page = pdf.pages[target_page - 1]  # pages are 0-indexed
            page_data = PageData(page_number=target_page)
            
            pdfplumber_metrics = measure_resource_usage(
                self._extract_pdfplumber_data,
                page, page_data
            )
            print(f"PDFPlumber Metrics:")
            print(f"Memory used: {pdfplumber_metrics['memory_delta']:.2f} MB")
            print(f"CPU usage: {pdfplumber_metrics['cpu_usage']}%")
            print(f"Time taken: {pdfplumber_metrics['time_taken']:.2f} seconds")
            
            page_data = pdfplumber_metrics['result']
            pages_data.append(page_data)
        
        # Step 2: Process with unstructured
        print("\n2. Processing with unstructured (fast strategy)...")
        try:
            elements = partition_pdf(
                pdf_path, 
                include_page_breaks=True,
                start_page=target_page,
                end_page=target_page,
                strategy="fast"  # Use fast strategy
            )
            
            # Process elements in parallel
            print(f"Using {max(1, multiprocessing.cpu_count() // 2)} CPU cores for element processing...")
            parallel_metrics = measure_resource_usage(
                process_elements_parallel,
                elements, target_page
            )
            print(f"Unstructured Processing Metrics:")
            print(f"Memory used: {parallel_metrics['memory_delta']:.2f} MB")
            print(f"CPU usage: {parallel_metrics['cpu_usage']}%")
            print(f"Time taken: {parallel_metrics['time_taken']:.2f} seconds")
            
            # Add processed elements to page data
            for element_text in parallel_metrics['result']:
                pages_data[0].text_blocks.append(element_text)
        except Exception as e:
            print(f"Warning: Unstructured processing failed with error: {e}")
            print("Continuing with PDFPlumber results only...")
        
        # Step 3: Process text with LLM in chunks
        print("\n3. Processing text with LLM in chunks...")
        for page_data in pages_data:
            if page_data.text_blocks:
                combined_text = " ".join(page_data.text_blocks)
                
                llm_metrics = measure_resource_usage(
                    self._process_text_with_llm,
                    combined_text
                )
                print(f"LLM Processing Metrics:")
                print(f"Memory used: {llm_metrics['memory_delta']:.2f} MB")
                print(f"CPU usage: {llm_metrics['cpu_usage']}%")
                print(f"Time taken: {llm_metrics['time_taken']:.2f} seconds")
                
                page_data.items.extend(llm_metrics['result'])
        
        print("\n=== PDF Processing Complete ===")
        return ExtractionResult(pages=pages_data)

    def _extract_pdfplumber_data(self, page, page_data: PageData) -> PageData:
        """Extract data using PDFPlumber"""
        try:
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:
                        headers = [str(h).strip() for h in table[0] if h is not None]
                        data = table[1:]
                        page_data.tables.append({
                            "headers": headers,
                            "data": data
                        })
            
            # Extract text
            text = page.extract_text()
            if text:
                page_data.text_blocks.append(text)
        except Exception as e:
            print(f"Error extracting data with PDFPlumber: {e}")
        
        return page_data

    def _parse_llm_response(self, response: str) -> List[PlumbingItem]:
        """Parse LLM response into PlumbingItem objects."""
        items = []
        for line in response.split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            try:
                fields = line.split('|')
                if len(fields) != 5:
                    continue
                
                item = PlumbingItem(
                    type=fields[0].strip(),
                    quantity=fields[1].strip(),
                    model_number=fields[2].strip(),
                    dimensions=fields[3].strip(),
                    mounting_type=fields[4].strip() if fields[4].strip() else None
                )
                items.append(item)
            except Exception as e:
                print(f"Error parsing item: {e}")
                continue
        
        return items 