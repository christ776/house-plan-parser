import os
import json
import pdfplumber
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from unstructured.partition.pdf import partition_pdf
import pandas as pd
from typing import List, Dict, Any
import torch
import gc
import psutil
import time
import sys
import requests
import re

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.Process(os.getpid()).cpu_percent()

def measure_resource_usage(func, *args, **kwargs):
    """Measure memory and CPU usage of a function"""
    start_memory = get_memory_usage()
    start_cpu = get_cpu_usage()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    end_cpu = get_cpu_usage()
    
    return {
        "memory_delta": end_memory - start_memory,
        "cpu_usage": end_cpu,
        "time_taken": end_time - start_time,
        "result": result
    }

def check_memory_usage(warning_threshold_mb=16000, limit_threshold_mb=20000):
    """Check if memory usage is approaching or exceeding limits"""
    current_memory = get_memory_usage()
    
    if current_memory > limit_threshold_mb:
        print(f"\nCRITICAL: Memory usage ({current_memory:.2f} MB) exceeds limit ({limit_threshold_mb} MB)")
        print("Stopping processing to prevent system issues")
        return False
    elif current_memory > warning_threshold_mb:
        print(f"\nWARNING: Memory usage ({current_memory:.2f} MB) is approaching limit ({limit_threshold_mb} MB)")
        print("Consider stopping or reducing processing load")
        return True
    return True

class PlumbingDataExtractor:
    def __init__(self, model_name="llama3"):
        self.output_dir = "extracted_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Ollama
        print("Initializing Ollama...")
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test Ollama connection
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": "Test connection",
                    "stream": False
                }
            )
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}")
            print(f"Successfully connected to Ollama using model: {self.model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Please ensure Ollama is running and the specified model is installed")
            print("You can install a model using: ollama pull llama2")
            sys.exit(1)

    def process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Process a single page"""
        current_memory = get_memory_usage()
        current_cpu = get_cpu_usage()
        print(f"\nProcessing page {page_num}...")
        print(f"Current memory usage: {current_memory:.2f} MB")
        print(f"Current CPU usage: {current_cpu}%")
        
        if not check_memory_usage():
            return None
        
        page_data = {
            "page_number": page_num,
            "tables": [],
            "text_blocks": [],
            "items": []
        }
        
        # Measure PDFPlumber operations
        print("\nMeasuring PDFPlumber operations...")
        pdfplumber_metrics = measure_resource_usage(
            self._extract_pdfplumber_data,
            page, page_num, page_data
        )
        print(f"PDFPlumber Metrics:")
        print(f"Memory used: {pdfplumber_metrics['memory_delta']:.2f} MB")
        print(f"CPU usage: {pdfplumber_metrics['cpu_usage']}%")
        print(f"Time taken: {pdfplumber_metrics['time_taken']:.2f} seconds")
        
        page_data = pdfplumber_metrics['result']
        
        # Process text with Ollama if we have text blocks
        if page_data["text_blocks"]:
            print("\nMeasuring LLM processing...")
            llm_metrics = measure_resource_usage(
                self._process_text_with_llm,
                page_data
            )
            print(f"LLM Processing Metrics:")
            print(f"Memory used: {llm_metrics['memory_delta']:.2f} MB")
            print(f"CPU usage: {llm_metrics['cpu_usage']}%")
            print(f"Time taken: {llm_metrics['time_taken']:.2f} seconds")
            
            page_data = llm_metrics['result']
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print final resource usage
        final_memory = get_memory_usage()
        final_cpu = get_cpu_usage()
        print(f"\nFinal metrics for page {page_num}:")
        print(f"Memory after cleanup: {final_memory:.2f} MB")
        print(f"Memory delta: {final_memory - current_memory:.2f} MB")
        print(f"Final CPU usage: {final_cpu}%")
        
        return page_data

    def _extract_pdfplumber_data(self, page, page_num: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data using PDFPlumber"""
        try:
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        try:
                            headers = [str(h).strip() for h in table[0] if h is not None]
                            data = table[1:]
                            
                            max_cols = max(len(row) for row in data)
                            cleaned_data = []
                            for row in data:
                                cleaned_row = row[:max_cols] + [None] * (max_cols - len(row))
                                cleaned_data.append(cleaned_row)
                            
                            df = pd.DataFrame(cleaned_data, columns=[f"col_{i}" for i in range(max_cols)])
                            table_dict = df.to_dict(orient="records")
                            del df
                            
                            page_data["tables"].append({
                                "table_index": table_idx,
                                "headers": headers,
                                "data": table_dict
                            })
                            
                            if not check_memory_usage():
                                return page_data
                                
                        except Exception as e:
                            print(f"Error processing table {table_idx} on page {page_num}: {e}")
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {e}")
        
        try:
            # Extract text
            text = page.extract_text()
            if text:
                page_data["text_blocks"].append(text)
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")
        
        return page_data

    def _process_text_with_llm(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with Ollama"""
        combined_text = " ".join(page_data["text_blocks"])
        try:
            # Split text into chunks
            max_chunk_size = 1000
            chunks = [combined_text[i:i+max_chunk_size] for i in range(0, len(combined_text), max_chunk_size)]
            
            all_items = []
            for chunk_idx, chunk in enumerate(chunks):
                print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                
                if not check_memory_usage():
                    page_data["items"] = all_items
                    return page_data
                
                # Simplified prompt with basic structure
                prompt = f"""Extract plumbing information from this text. For each item, provide:
                - type (pipe, fitting, valve, etc.)
                - quantity (number)
                - model_number (system code like HHWS, CWR)
                - dimensions (size and length)
                - mounting_type (if specified)
                
                Text: {chunk}
                
                Format each item as a simple line with fields separated by |:
                type|quantity|model_number|dimensions|mounting_type
                
                Example:
                pipe|2|HHWS|3/4 inch|25 ft -0 5/8
                fitting|1|CD|1/2 inch|
                
                Now extract the plumbing information and return ONLY the items in this format:
                """
                
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        response = requests.post(
                            self.ollama_url,
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "top_k": 40
                                }
                            }
                        )
                        
                        if response.status_code != 200:
                            raise Exception(f"Ollama API returned status code {response.status_code}")
                        
                        response_data = response.json()
                        response_text = response_data.get("response", "").strip()
                        
                        # Process each line of the response
                        items = []
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if not line or '|' not in line:
                                continue
                                
                            try:
                                # Split the line into fields
                                fields = line.split('|')
                                if len(fields) != 5:
                                    continue
                                    
                                # Create item dictionary
                                item = {
                                    "type": fields[0].strip().lower(),
                                    "quantity": fields[1].strip(),
                                    "model_number": fields[2].strip(),
                                    "dimensions": fields[3].strip(),
                                    "mounting_type": fields[4].strip()
                                }
                                
                                # Clean up the data
                                # Replace special characters
                                item["dimensions"] = item["dimensions"].replace("ø", "inch")
                                item["dimensions"] = item["dimensions"].replace("\'", "ft")
                                item["dimensions"] = item["dimensions"].replace("\"", "inch")
                                
                                # Clean up model number
                                item["model_number"] = re.sub(r'\s*inch\s*', '', item["model_number"])
                                
                                # Determine type if empty
                                if not item["type"]:
                                    if item["model_number"] in ["HHWS", "HHWR", "CWS", "CWR", "CHWS", "CHWR"]:
                                        item["type"] = "pipe"
                                    elif item["model_number"] in ["CD"]:
                                        item["type"] = "fitting"
                                
                                # Clean up quantity and dimensions
                                if not item["quantity"] and item["dimensions"]:
                                    dim_parts = item["dimensions"].split()
                                    if dim_parts and any(c.isdigit() for c in dim_parts[0]):
                                        item["quantity"] = dim_parts[0]
                                        item["dimensions"] = " ".join(dim_parts[1:])
                                
                                # Ensure model_number is uppercase for system codes
                                if item["model_number"] in ["HHWS", "HHWR", "CWS", "CWR", "CHWS", "CHWR", "CD"]:
                                    item["model_number"] = item["model_number"].upper()
                                
                                # Post-process the data
                                # Fix double "inch" in dimensions
                                item["dimensions"] = re.sub(r'inchinch', 'inch', item["dimensions"])
                                
                                # Standardize spacing around "inch"
                                item["dimensions"] = re.sub(r'(\d+)\s*(inch)', r'\1 \2', item["dimensions"])
                                item["dimensions"] = re.sub(r'(inch)\s*(\d+)', r'\1 \2', item["dimensions"])
                                
                                # Handle CD in dimensions
                                if "CD" in item["dimensions"]:
                                    item["model_number"] = "CD"
                                    item["type"] = "fitting"
                                    item["dimensions"] = item["dimensions"].replace("CD", "").strip()
                                
                                # Standardize mounting type format
                                if item["mounting_type"]:
                                    # Replace ' with ft and " with inch
                                    item["mounting_type"] = item["mounting_type"].replace("'", "ft")
                                    item["mounting_type"] = item["mounting_type"].replace('"', "inch")
                                    
                                    # Standardize spacing in mounting type
                                    item["mounting_type"] = re.sub(r'(\d+)\s*(ft|inch)', r'\1 \2', item["mounting_type"])
                                    item["mounting_type"] = re.sub(r'(ft|inch)\s*(\d+)', r'\1 \2', item["mounting_type"])
                                    
                                    # Handle multiple mounting types
                                    if "," in item["mounting_type"]:
                                        mount_types = [m.strip() for m in item["mounting_type"].split(",")]
                                        for mount_type in mount_types[1:]:
                                            new_item = item.copy()
                                            new_item["mounting_type"] = mount_type
                                            items.append(new_item)
                                        item["mounting_type"] = mount_types[0]
                                
                                # Create a unique key for deduplication
                                dedup_key = (
                                    item["type"],
                                    item["quantity"],
                                    item["model_number"],
                                    item["dimensions"],
                                    item["mounting_type"]
                                )
                                
                                # Add the item if it's not a duplicate
                                if dedup_key not in {(i["type"], i["quantity"], i["model_number"], i["dimensions"], i["mounting_type"]) for i in items}:
                                    items.append(item)
                                
                            except Exception as e:
                                print(f"Error processing line: {line}")
                                print(f"Error: {str(e)}")
                                continue
                        
                        if items:
                            all_items.extend(items)
                            print(f"Successfully processed chunk {chunk_idx + 1}")
                            break  # Success, exit retry loop
                        else:
                            raise ValueError("No valid items found in response")
                        
                    except Exception as e:
                        print(f"Attempt {retry + 1}/{max_retries} failed: {str(e)}")
                        if retry == max_retries - 1:
                            print("Max retries reached, skipping chunk")
                            print("Problematic response was:", response_text[:200] + "..." if len(response_text) > 200 else response_text)
                        else:
                            print("Retrying...")
                            time.sleep(1)  # Brief pause before retry
                
                # Clear memory after each chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if not check_memory_usage():
                    page_data["items"] = all_items
                    return page_data
            
            # Final deduplication and cleanup pass
            unique_items = []
            seen_keys = set()
            for item in all_items:
                # Fix CD pipes that were incorrectly converted
                if "CD" in item["dimensions"] or item["model_number"] == "CD":
                    item["model_number"] = "CD"
                    item["type"] = "fitting"
                    item["dimensions"] = item["dimensions"].replace("CD", "").strip()
                
                # Fix valve and fitting types
                if item["model_number"] in ["M7"]:
                    item["type"] = "valve"
                elif item["model_number"] == "CD":
                    item["type"] = "fitting"
                
                # Standardize "inches" to "inch"
                item["dimensions"] = item["dimensions"].replace("inches", "inch")
                
                # Fix spacing in mounting type
                if item["mounting_type"]:
                    item["mounting_type"] = re.sub(r'(\d+)\s*(ft|inch)', r'\1 \2', item["mounting_type"])
                    item["mounting_type"] = re.sub(r'(ft|inch)\s*(\d+)', r'\1 \2', item["mounting_type"])
                
                # Create deduplication key
                key = (
                    item["type"],
                    item["quantity"],
                    item["model_number"],
                    item["dimensions"],
                    item["mounting_type"]
                )
                
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_items.append(item)
            
            page_data["items"] = unique_items
        except Exception as e:
            print(f"Error in LLM processing: {e}")
        
        return page_data

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text and structure from PDF"""
        extracted_data = []
        start_time = time.time()
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages in PDF: {total_pages}")
            
            # Process only page 4
            target_page = 4
            if target_page > total_pages:
                print(f"Error: PDF only has {total_pages} pages, cannot process page {target_page}")
                return extracted_data
                
            print(f"\nProcessing only page {target_page}...")
            page = pdf.pages[target_page - 1]  # pages are 0-indexed
            
            # Check memory before processing
            if not check_memory_usage():
                print("Memory limit reached. Stopping processing.")
                return extracted_data
            
            page_data = self.process_page(page, target_page)
            if page_data is None:
                print("Memory limit reached during page processing. Stopping.")
                return extracted_data
                
            extracted_data.append(page_data)
            
            # Save results for page 4
            output_path = os.path.join(self.output_dir, f"{os.path.basename(pdf_path)}_page_{target_page}.json")
            with open(output_path, 'w') as f:
                json.dump(page_data, f, indent=2)
            print(f"Saved results for page {target_page}")
        
        # Method 2: Using unstructured for layout-aware extraction (only for page 4)
        print("\nProcessing with unstructured...")
        try:
            elements = partition_pdf(pdf_path, include_page_breaks=True)
            for element in elements:
                if hasattr(element, "page_number") and element.page_number == target_page:
                    extracted_data[0]["text_blocks"].append(str(element))
                    
                    # Check memory after each element
                    if not check_memory_usage():
                        print("Memory limit reached during unstructured processing. Stopping.")
                        return extracted_data
        except Exception as e:
            print(f"Error in unstructured processing: {e}")
        
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        return extracted_data

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main method to process a PDF and extract structured data"""
        print(f"\nStarting processing of {pdf_path}")
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Check initial memory
        if not check_memory_usage():
            print("Initial memory usage too high. Exiting.")
            return None
        
        # Extract raw data
        extracted_data = self.extract_text_from_pdf(pdf_path)
        
        if extracted_data:
            # Save final results
            output_path = os.path.join(self.output_dir, f"{os.path.basename(pdf_path)}_extracted.json")
            with open(output_path, 'w') as f:
                json.dump(extracted_data, f, indent=2)
        
        final_memory = get_memory_usage()
        print(f"\nFinal memory usage: {final_memory:.2f} MB")
        print(f"Total memory used: {final_memory - initial_memory:.2f} MB")
        
        return extracted_data

def main():
    extractor = PlumbingDataExtractor()
    pdfs_dir = "pdfs"
    
    # Process all PDFs in the directory
    for pdf_file in os.listdir(pdfs_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdfs_dir, pdf_file)
            print(f"\nProcessing {pdf_file}...")
            results = extractor.process_pdf(pdf_path)
            
            if results is None:
                print(f"Processing stopped for {pdf_file} due to memory constraints")
                continue
            
            # Print summary of extracted data
            total_items = sum(len(page["items"]) for page in results)
            print(f"\nExtraction Summary for {pdf_file}:")
            print(f"Total pages processed: {len(results)}")
            print(f"Total items found: {total_items}")
            
            # Print details for each page
            for page in results:
                if page["items"]:
                    print(f"\nPage {page['page_number']}:")
                    print(f"Found {len(page['items'])} items")
                    for item in page["items"]:
                        print(f"- {item['type']}: {item.get('model_number', 'N/A')} "
                              f"(Qty: {item.get('quantity', 'N/A')}, "
                              f"Dim: {item.get('dimensions', 'N/A')}, "
                              f"Mount: {item.get('mounting_type', 'N/A')})")
            
            print(f"\nDetailed results saved to {pdf_file}_extracted.json")
            
            # Optionally save a summary CSV
            summary_data = []
            for page in results:
                for item in page["items"]:
                    summary_data.append({
                        "page": page["page_number"],
                        **item
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_path = os.path.join(extractor.output_dir, f"{os.path.basename(pdf_file)}_summary.csv")
                df.to_csv(csv_path, index=False)
                print(f"Summary CSV saved to {csv_path}")

if __name__ == "__main__":
    main() 