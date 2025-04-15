from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..models.plumbing import PlumbingItem, PageData, ExtractionResult
import re

class ValidationChain:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        self.memory = ConversationBufferMemory()
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=PlumbingItem)
        
        # Initialize caches
        self.type_cache = {}  # Cache for type conversions
        self.model_cache = {}  # Cache for model number conversions
        self.dimension_cache = {}  # Cache for dimension standardization
        self.mounting_cache = {}  # Cache for mounting type standardization
        
        # Create validation prompt
        self.validation_prompt = PromptTemplate(
            input_variables=["item"],
            template="""Standardize this plumbing item. Follow these exact rules:

1. Types (must be one of): pipe, valve, fitting, fixture, accessory
   - Convert to singular form (e.g., "pipes" → "pipe")
   - Convert "condensate" or "condensate pipe" to "pipe"
   - Convert "valve package" or "isolation valve" to "valve"
   - "CD" is NOT a valid type, it should be "pipe"

2. Model Numbers (must be one of): HHWS, HHWR, CWS, CWR, CHWS, CHWR, CD, M7
   - Convert "HHHS" or "HHS" to "HHWS"
   - Convert "CWSR" to "CWS"
   - Convert "CHW" to "CHWS"
   - Convert "CHWRR" to "CHWR"
   - Convert "CONDENSATE" or variations to "CWR"
   - Convert "VALVE PACKAGE" or "ISOLATION VALVE" to "HHWS"
   - Convert "PILOT", "DWVW", or "CPVC" to "HHWS"
   - For pipes with invalid model numbers (including "CD" and "M7"), use blank ("")
   - Remove any quotes or brackets around model numbers
   - Remove any special characters from model numbers

3. Dimensions Format: "X inch" or "X ft - Y inch"
   - Remove "ø" symbol
   - Convert "inches" to "inch"
   - Convert " to "inch"
   - Convert ' to "ft"
   - Always use "ft -" (with space and hyphen)
   - Keep fractions as is (e.g., "1/2 inch", "3/4 inch")
   - Do not modify valid dimensions
   - For dimensions with "x" separator:
     * If both parts have units, keep both (e.g., "3 1/4 inch x 10 ft")
     * If only one part has units, add units to the other (e.g., "3 1/4 inch x 10 ft")
   - Convert hyphenated fractions (e.g., "1-1/4 inch" → "1 1/4 inch")
   - Remove any quotes around dimensions
   - If multiple dimensions are given, use only the first valid one
   - Ensure consistent spacing around hyphens
   - Always include "inch" suffix

4. Mounting Type Format: "X ft - Y inch"
   - Convert " to "inch"
   - Convert ' to "ft"
   - Always use "ft -" (with space and hyphen)
   - Keep fractions as is
   - Always include "inch" suffix
   - Do not modify valid mounting types
   - For mounting types with "BE=" prefix:
     * Remove "BE=" and any extra spaces
     * Keep the rest of the mounting type as is
   - For mounting types with "up to" prefix:
     * Remove "up to" and any extra spaces
     * Keep the rest of the mounting type as is
   - For mounting types with multiple values:
     * Split by comma and process each part separately
     * Create a separate item for each mounting type
     * Keep the original quantity for each split item
   - Remove any quotes around mounting types
   - Remove any duplicate "inch" suffixes
   - Ensure consistent spacing around hyphens
   - Remove any leading or trailing spaces
   - If no mounting type is specified, use blank ("")
   - Always include "inch" suffix

Item: {item}

Return ONLY the standardized item in this exact format:
type|quantity|model_number|dimensions|mounting_type

Do not include any explanations or additional text.
"""
        )
        
        # Create validation chain using RunnableSequence
        self.validation_chain = RunnableSequence(
            self.validation_prompt | self.llm
        )

    def validate_items(self, items: List[PlumbingItem]) -> List[PlumbingItem]:
        """Validate and correct a list of plumbing items."""
        validated_items = []
        seen_items = set()
        
        for item in items:
            try:
                # Create a string representation of the item
                item_str = f"{item.type}|{item.quantity}|{item.model_number}|{item.dimensions}|{item.mounting_type or ''}"
                
                # Get validation response
                response = self.validation_chain.invoke({"item": item_str})
                
                # Parse the response
                validated_item = self._parse_validation_response(response)
                if validated_item:
                    # Create deduplication key
                    dedup_key = (
                        validated_item.type,
                        validated_item.quantity,
                        validated_item.model_number,
                        validated_item.dimensions,
                        validated_item.mounting_type
                    )
                    
                    # Add if not a duplicate
                    if dedup_key not in seen_items:
                        seen_items.add(dedup_key)
                        validated_items.append(validated_item)
            
            except Exception as e:
                print(f"Error validating item: {e}")
                continue
        
        return validated_items

    def _parse_validation_response(self, response: str) -> PlumbingItem:
        """Parse validation response into a PlumbingItem object."""
        try:
            # Split the response into fields
            fields = response.split('|')
            if len(fields) != 5:
                return None
            
            # Create and return the validated item
            return PlumbingItem(
                type=fields[0].strip(),
                quantity=fields[1].strip(),
                model_number=fields[2].strip(),
                dimensions=fields[3].strip(),
                mounting_type=fields[4].strip() if fields[4].strip() else None
            )
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return None

    def _process_text_with_llm(self, text: str) -> List[PlumbingItem]:
        """Process text with LLM in chunks"""
        max_chunk_size = 1000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        all_items = []
        for chunk in chunks:
            try:
                # Create extraction prompt with clear guidance
                extraction_prompt = f"""Extract plumbing information from this text. For each item, provide:
                - type (must be one of: pipe, fitting, valve, fixture, accessory)
                - quantity (number)
                - model_number (must be one of: HHWS, HHWR, CWR, CWS, CHWS, CHWR, CD, M7)
                - dimensions (size in inches, e.g., "3/4 inch", "1 1/2 inch")
                - mounting_type (if specified, in format "X ft - Y Z inch" or "X' - Y Z\"" or similar)
                
                Text: {chunk}
                
                Format each item as a simple line with fields separated by |:
                type|quantity|model_number|dimensions|mounting_type
                
                Examples:
                pipe|2|HHWS|3/4 inch|25 ft -0 5/8 inch
                pipe|1|HHWR|1 1/2 inch|15 ft -0 inch
                pipe|1|CWR|6 inch|27 ft -2 7/8 inch
                
                Return ONLY the items in this format, one per line.
                Do not include any explanations or additional text.
                """
                
                # Use the new RunnableSequence format
                response = self.extraction_chain.invoke({"text": extraction_prompt})
                items = self._parse_llm_response(response)
                all_items.extend(items)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        return all_items

    def validate_extraction_result(self, result: ExtractionResult) -> ExtractionResult:
        """Validate the extracted data using LLM."""
        validated_pages = []
        
        for page in result.pages:
            if not page.items:
                validated_pages.append(page)
                continue
                
            # Convert items to text format for batch processing
            items_text = "\n".join([
                f"{item.type}|{item.quantity}|{item.model_number}|{item.dimensions}|{item.mounting_type or ''}"
                for item in page.items
            ])
            
            # Create batch validation prompt
            batch_prompt = f"""Validate and correct these plumbing items. Check for:
            1. Correct type (must be one of: pipe, fitting, valve, fixture, accessory)
            2. Valid model number (must be one of: HHWS, HHWR, CWR, CWS, CHWS, CHWR, CD, M7)
            3. Proper dimension formatting (use inches, e.g., "3/4 inch", "1 1/2 inch")
            4. Valid mounting type if specified (in format "X ft - Y Z inch")
            
            For model numbers:
            - If a pipe has no model number, use HHWS for hot water supply or HHWR for hot water return
            - If a valve has no model number, use HHWS
            - For condensate pipes, use CWR
            - For chilled water, use CHWS or CHWR
            
            Items:
            {items_text}
            
            Return ONLY the corrected items in the exact format below, one per line:
            type|quantity|model_number|dimensions|mounting_type
            
            If an item is valid, return it as is. If invalid, return the corrected version.
            Do not include any explanations or additional text.
            """
            
            try:
                # Process all items in one batch
                response = self.llm.invoke(batch_prompt)
                validated_items = self._parse_llm_response(response)
                
                # If validation failed to return any items, keep the original items
                if not validated_items:
                    print(f"Warning: Validation returned no items for page {page.page_number}, keeping original items")
                    validated_items = page.items
                
                # Create new page with validated items
                validated_page = PageData(
                    page_number=page.page_number,
                    items=validated_items,
                    text_blocks=page.text_blocks,
                    tables=page.tables
                )
                validated_pages.append(validated_page)
                
            except Exception as e:
                print(f"Error validating page {page.page_number}: {e}")
                validated_pages.append(page)  # Keep original items if validation fails
        
        return ExtractionResult(pages=validated_pages)

    def _standardize_with_cache(self, value: str, cache: dict, field_type: str) -> str:
        """Standardize a value using cache and LLM if needed."""
        if not value:
            return value
            
        # Check cache first
        if value in cache:
            return cache[value]
            
        # If not in cache, use LLM to standardize
        try:
            # Create a simple item with just the field we want to standardize
            item = f"dummy|1|HHWS|1 inch|1 ft - 1 inch"
            if field_type == "type":
                item = f"{value}|1|HHWS|1 inch|1 ft - 1 inch"
            elif field_type == "model":
                item = f"pipe|1|{value}|1 inch|1 ft - 1 inch"
            elif field_type == "dimension":
                item = f"pipe|1|HHWS|{value}|1 ft - 1 inch"
            elif field_type == "mounting":
                item = f"pipe|1|HHWS|1 inch|{value}"
                
            response = self.validation_chain.invoke({"item": item})
            standardized = response.split('|')[0 if field_type == "type" else 
                                            2 if field_type == "model" else 
                                            3 if field_type == "dimension" else 
                                            4]
            
            # Add to cache
            cache[value] = standardized
            return standardized
        except Exception:
            # For model numbers, if standardization fails, try to validate against known values
            if field_type == "model" and value.upper() in {'HHWS', 'HHWR', 'CWS', 'CWR', 'CHWS', 'CHWR', 'CD', 'M7'}:
                return value.upper()
            return value

    def _standardize_dimensions(self, dimensions: str) -> str:
        """Standardize dimension format."""
        if not dimensions:
            return dimensions
            
        # Remove any special characters and standardize format
        dimensions = dimensions.replace('"', ' inch').replace("'", ' inch')
        dimensions = dimensions.replace('ø', '').strip()
        dimensions = dimensions.replace('inches', 'inch')  # Standardize to singular
        dimensions = dimensions.replace('  ', ' ')  # Remove double spaces
        
        # Handle decimal inches
        if '.' in dimensions:
            parts = dimensions.split('.')
            if len(parts) == 2:
                whole = parts[0]
                fraction = parts[1]
                if fraction == '5':
                    dimensions = f"{whole} 1/2 inch"
                elif fraction == '25':
                    dimensions = f"{whole} 1/4 inch"
                elif fraction == '75':
                    dimensions = f"{whole} 3/4 inch"
                else:
                    dimensions = f"{whole} {fraction}/100 inch"
        
        # Handle hyphenated fractions (e.g., "1-1/4 inch" → "1 1/4 inch")
        if '-' in dimensions and '/' in dimensions:
            parts = dimensions.split('-')
            if len(parts) == 2:
                whole = parts[0].strip()
                fraction = parts[1].strip()
                dimensions = f"{whole} {fraction}"
        
        # Handle feet and inches format
        if "'" in dimensions or '"' in dimensions:
            parts = dimensions.split()
            if len(parts) >= 2:
                feet = parts[0].replace("'", "").replace('"', "")
                inches = " ".join(parts[1:])
                dimensions = f"{feet} ft - {inches}"
        
        # Ensure "inch" is present
        if 'inch' not in dimensions and not dimensions.endswith('ft'):
            dimensions = f"{dimensions} inch"
        
        # Fix any remaining formatting issues
        dimensions = dimensions.replace('  ', ' ').strip()
        return dimensions

    def _standardize_mounting_type(self, mounting_type: str) -> str:
        """Standardize mounting type format."""
        if not mounting_type:
            return mounting_type
            
        # Remove any special characters and standardize format
        mounting_type = mounting_type.replace('"', ' inch').replace("'", ' inch')
        mounting_type = mounting_type.replace('BE=', '').strip()
        
        # Convert feet/inches format
        if "'" in mounting_type or '"' in mounting_type:
            parts = mounting_type.split()
            if len(parts) >= 2:
                feet = parts[0].replace("'", "").replace('"', "")
                inches = " ".join(parts[1:])
                mounting_type = f"{feet} ft - {inches}"
        
        # Handle decimal feet
        if '.' in mounting_type:
            parts = mounting_type.split('.')
            if len(parts) == 2:
                whole = parts[0]
                fraction = parts[1]
                if fraction == '5':
                    mounting_type = f"{whole} 1/2 inch"
                elif fraction == '25':
                    mounting_type = f"{whole} 1/4 inch"
                elif fraction == '75':
                    mounting_type = f"{whole} 3/4 inch"
                else:
                    mounting_type = f"{whole} {fraction}/100 inch"
        
        # Ensure proper spacing around hyphens
        mounting_type = mounting_type.replace('ft-', 'ft - ')
        mounting_type = mounting_type.replace('  ', ' ')
        
        # Handle multiple mounting types
        if ',' in mounting_type:
            mount_types = [m.strip() for m in mounting_type.split(',')]
            mounting_type = ', '.join([self._standardize_mounting_type(m) for m in mount_types])
        
        return mounting_type

    def _parse_llm_response(self, response: str) -> List[PlumbingItem]:
        """Parse LLM response into PlumbingItem objects."""
        items = []
        seen_items = set()  # Track unique items to avoid duplicates
        
        # Split response into lines and look for lines containing pipe-separated values
        for line in response.split('\n'):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            
            # Find the pipe-separated part of the line
            pipe_part = line
            if ':' in line:
                # If there's a colon, take the part after it
                pipe_part = line.split(':')[-1].strip()
            
            # Skip if no pipe separator
            if '|' not in pipe_part:
                continue
            
            # Split by pipe and ensure we have exactly 5 parts
            fields = pipe_part.split('|')
            if len(fields) != 5:
                continue
            
            try:
                # Clean up each field
                type_ = self._standardize_with_cache(fields[0].strip().lower(), self.type_cache, "type")
                quantity = fields[1].strip()
                model_number = self._standardize_with_cache(fields[2].strip().upper(), self.model_cache, "model")
                dimensions = self._standardize_dimensions(fields[3].strip())
                mounting_type = fields[4].strip() if fields[4].strip() else None
                
                # Handle missing or invalid quantities
                if not quantity or quantity.lower() in ['na', '?', 'none', '', '--']:
                    quantity = '1'  # Default to 1 if quantity is missing
                
                # Skip items with empty required fields (except quantity which we handle above)
                if not all([type_, dimensions]):
                    continue
                
                # Handle multiple dimensions
                if ',' in dimensions:
                    # Split by comma and process each dimension separately
                    dim_parts = [d.strip() for d in dimensions.split(',')]
                    for dim in dim_parts:
                        # Standardize dimension
                        std_dim = self._standardize_dimensions(dim)
                        # Standardize mounting type if present
                        std_mount = self._standardize_mounting_type(mounting_type) if mounting_type else None
                        
                        # Create a unique key for deduplication
                        item_key = (type_, quantity, model_number, std_dim, std_mount)
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            item = PlumbingItem(
                                type=type_,
                                quantity=quantity,
                                model_number=model_number,
                                dimensions=std_dim,
                                mounting_type=std_mount
                            )
                            items.append(item)
                else:
                    # Single dimension
                    # Standardize dimension and mounting type
                    std_dim = self._standardize_dimensions(dimensions)
                    std_mount = self._standardize_mounting_type(mounting_type) if mounting_type else None
                    
                    # Create a unique key for deduplication
                    item_key = (type_, quantity, model_number, std_dim, std_mount)
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        item = PlumbingItem(
                            type=type_,
                            quantity=quantity,
                            model_number=model_number,
                            dimensions=std_dim,
                            mounting_type=std_mount
                        )
                        items.append(item)
            except Exception as e:
                print(f"Warning: Error parsing item '{line}': {str(e)}")
                continue
        
        return items 