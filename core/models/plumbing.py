from typing import Optional, List
from pydantic import BaseModel, Field, validator
import re

class PlumbingItem(BaseModel):
    """Model for a single plumbing item with validation rules."""
    type: str = Field(..., description="Type of plumbing item (pipe, fitting, valve, etc.)")
    quantity: str = Field(..., description="Quantity of the item")
    model_number: str = Field(..., description="System code like HHWS, CWR")
    dimensions: str = Field(..., description="Size and length specifications")
    mounting_type: Optional[str] = Field(None, description="Mounting specifications if any")

    @validator('type')
    def validate_type(cls, v):
        valid_types = {'pipe', 'fitting', 'valve', 'fixture', 'accessory'}
        v = v.lower().strip()
        # Remove any special characters and standardize
        v = re.sub(r'[^a-zA-Z]', '', v)
        if v not in valid_types:
            # Try to find a close match
            for valid_type in valid_types:
                if valid_type in v or v in valid_type:
                    return valid_type
            # Default to pipe if no match found
            return 'pipe'
        return v

    @validator('model_number')
    def validate_model_number(cls, v):
        valid_models = {'HHWS', 'HHWR', 'CWS', 'CWR', 'CHWS', 'CHWR', 'CD', 'M7'}
        v = v.upper().strip()
        if not v or v == 'N/A':  # Handle empty or N/A model numbers
            return ''  # Return empty string instead of defaulting
        # Remove any special characters and standardize
        v = re.sub(r'[^A-Z0-9]', '', v)
        if v not in valid_models:
            # Try to find a close match
            for model in valid_models:
                if model in v or v in model:
                    return model
            return ''  # Return empty string if no match found
        return v

    @validator('dimensions')
    def validate_dimensions(cls, v):
        if not v:
            return v
        # Standardize dimension format
        v = v.replace("ø", "inch")
        v = v.replace("\'", "ft")
        v = v.replace("\"", "inch")
        v = re.sub(r'inchinch', 'inch', v)
        v = re.sub(r'(\d+)\s*(inch)', r'\1 \2', v)
        v = re.sub(r'(inch)\s*(\d+)', r'\1 \2', v)
        # Handle BE= prefix
        v = v.replace("BE=", "").strip()
        # Handle hyphenated fractions
        v = re.sub(r'(\d+)-(\d+/\d+)', r'\1 \2', v)
        # Ensure proper spacing around hyphens
        v = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 - \2', v)
        return v.strip()

    @validator('mounting_type')
    def validate_mounting_type(cls, v):
        if v is None:
            return v
        # Standardize mounting type format
        v = v.replace("'", "ft")
        v = v.replace('"', "inch")
        v = v.replace("BE=", "").strip()
        v = re.sub(r'(\d+)\s*(ft|inch)', r'\1 \2', v)
        v = re.sub(r'(ft|inch)\s*(\d+)', r'\1 \2', v)
        # Handle hyphenated fractions
        v = re.sub(r'(\d+)-(\d+/\d+)', r'\1 \2', v)
        # Ensure proper spacing around hyphens
        v = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 - \2', v)
        return v.strip()

class PageData(BaseModel):
    """Model for data extracted from a single page."""
    page_number: int
    tables: List[dict] = Field(default_factory=list)
    text_blocks: List[str] = Field(default_factory=list)
    items: List[PlumbingItem] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    """Model for the complete extraction result."""
    pages: List[PageData]
    metadata: dict = Field(default_factory=dict) 