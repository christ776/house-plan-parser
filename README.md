# House Plan Parser

A Python-based tool for extracting and standardizing plumbing information from house plans and construction documents.

## Features

- Extracts plumbing information from PDF documents
- Standardizes plumbing item formats
- Handles various input formats and notations
- Validates extracted data against predefined rules
- Supports parallel processing for improved performance

## Architecture Overview

The system is built using a modular architecture with the following components:

### Core Components

1. **PDF Processing Module**

   - Uses `pdfplumber` for basic PDF text extraction
   - Uses `unstructured` for layout-aware text extraction
   - Supports parallel processing for improved performance

2. **Extraction Chain**

   - Uses LangChain for structured LLM workflows
   - Processes text in chunks to handle large documents
   - Extracts and standardizes plumbing information
   - Handles various input formats and notations

3. **Validation Chain**

   - Validates extracted data against predefined rules
   - Standardizes model numbers, dimensions, and mounting types
   - Handles special cases and edge conditions
   - Ensures consistent output format

4. **Data Models**
   - Uses Pydantic models for type-safe data validation
   - Defines standard formats for plumbing items
   - Enforces validation rules for each field
   - Handles error cases gracefully

### Workflow

```mermaid
graph TD
    A[PDF Document] --> B[PDF Processing]
    B --> C[Text Extraction]
    C --> D[Chunk Processing]
    D --> E[LLM Extraction]
    E --> F[Data Validation]
    F --> G[Standardized Output]
    G --> H[JSON/CSV Files]

    subgraph "PDF Processing"
    B --> B1[pdfplumber]
    B --> B2[unstructured]
    end

    subgraph "Extraction & Validation"
    E --> E1[Model Number]
    E --> E2[Dimensions]
    E --> E3[Mounting Type]
    E --> E4[Quantity]
    end

    subgraph "Output Generation"
    G --> G1[extracted.json]
    G --> G2[summary.csv]
    end
```

### Processing Steps

1. **PDF Processing**

   - Document is loaded and preprocessed
   - Text is extracted using layout-aware methods
   - Pages are processed in parallel for efficiency

2. **Text Processing**

   - Text is split into manageable chunks
   - Each chunk is processed by the LLM
   - Initial items are extracted and formatted

3. **Data Extraction**

   - Model numbers are extracted and standardized
   - Dimensions are converted to standard format
   - Mounting types are validated and formatted
   - Quantities are standardized

4. **Validation**

   - Each field is validated against rules
   - Special cases are handled
   - Invalid data is corrected or removed
   - Duplicates are eliminated

5. **Output Generation**
   - Validated items are formatted
   - Results are saved to JSON and CSV
   - Summary statistics are generated

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Using uv (Recommended)

1. Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/house_plan_parser.git
cd house_plan_parser
```

3. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Using pip

1. Clone the repository:

```bash
git clone https://github.com/yourusername/house_plan_parser.git
cd house_plan_parser
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
uv sync
```

### Dependencies

The project uses the following main dependencies:

- `pdfplumber`: For PDF text extraction
- `unstructured`: For layout-aware text extraction
- `langchain`: For LLM workflows
- `pydantic`: For data validation
- `pandas`: For data processing and CSV output

### Development Setup

For development, install additional dependencies:

```bash
uv pip install -r requirements-dev.txt
```

## Example Output

### Processing Metrics

The system provides detailed metrics about the processing performance:

```
LLM Processing Metrics:
Memory used: 0.12 MB
CPU usage: 86.0%
Time taken: 108.07 seconds

=== PDF Processing Complete ===
```

### Extracted Data Examples

The system extracts and standardizes plumbing information into a consistent format:

```
type|quantity|model_number|dimensions|mounting_type
pipe|1|HHWS|19ft - 8 1/4 inch|-
pipe|1|HHWS|14ft - 1 inch|-
pipe|1|HHWS|9ft - 0 1/8 inch|-
pipe|1|HHWS|12ft - 9 inch|-
pipe|1|HHWS|12ft - 8 3/4 inch|-
pipe|1|HHWS|19ft - 3 3/4 inch|-
pipe|1|HHWS|19ft - 3 1/4 inch|-
pipe|1|HHWS|25ft - 1 inch|-
pipe|1|HHWS|26ft - 7 1/8 inch|-
pipe|1|HHWS|15ft - 5 3/8 inch|-
```

### Summary Output

The system generates a summary of the extracted data:

```
Extraction Summary for M&P mark-up against shop systems piping.pdf:
Total pages processed: 1
Total items found: 20

Page 4:
Found 20 items
- pipe: (Qty: 1, Dim: 19ft 8 1/4 inch, Mount: -)
- pipe: (Qty: 1, Dim: 14ft - 1 inch, Mount: -)
- pipe: (Qty: 1, Dim: 9ft 0 1/8 inch, Mount: -)
...
```

### Output Files

The system generates two types of output files:

1. `extracted_data/[filename]_extracted.json` - Detailed JSON data
2. `extracted_data/[filename]_summary.csv` - Summary in CSV format
