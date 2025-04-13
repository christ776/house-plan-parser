# �� Imports
import os
import json
from typing import List
import pandas as pd
from workflows.plumbing import run_workflow
from core.models.plumbing import ExtractionResult

def process_pdf(pdf_path: str, model_name: str = "llama3") -> ExtractionResult:
    """Process a single PDF file using the plumbing workflow."""
    print(f"\nProcessing {pdf_path}...")
    try:
        result = run_workflow(pdf_path, model_name)
        return result
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def save_results(result: ExtractionResult, pdf_path: str, output_dir: str = "extracted_data"):
    """Save extraction results to JSON and CSV files with detailed page information."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare detailed JSON output
    detailed_output = {
        "source_file": os.path.basename(pdf_path),
        "total_pages": len(result.pages),
        "total_items": sum(len(page.items) for page in result.pages),
        "pages": []
    }
    
    # Process each page
    for page in result.pages:
        page_data = {
            "page_number": page.page_number,
            "total_items": len(page.items),
            "items": []
        }
        
        # Process each item on the page
        for item in page.items:
            item_data = {
                "type": item.type,
                "quantity": item.quantity,
                "model_number": item.model_number,
                "dimensions": item.dimensions,
                "mounting_type": item.mounting_type,
                "page_number": page.page_number,
                "source_file": os.path.basename(pdf_path)
            }
            page_data["items"].append(item_data)
        
        detailed_output["pages"].append(page_data)
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_extracted.json")
    with open(json_path, 'w') as f:
        json.dump(detailed_output, f, indent=2)
    print(f"Saved detailed results to {json_path}")
    
    # Prepare CSV data
    csv_data = []
    for page in result.pages:
        for item in page.items:
            csv_data.append({
                "page_number": page.page_number,
                "type": item.type,
                "quantity": item.quantity,
                "model_number": item.model_number,
                "dimensions": item.dimensions,
                "mounting_type": item.mounting_type,
                "source_file": os.path.basename(pdf_path)
            })
    
    # Save summary CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        # Reorder columns for better readability
        df = df[["source_file", "page_number", "type", "quantity", "model_number", "dimensions", "mounting_type"]]
        csv_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved summary to {csv_path}")

def main():
    # Configuration
    pdfs_dir = "pdfs"
    model_name = "llama3"
    
    # Process all PDFs in the directory
    for pdf_file in os.listdir(pdfs_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdfs_dir, pdf_file)
            result = process_pdf(pdf_path, model_name)
            
            if result is None:
                print(f"Processing stopped for {pdf_file}")
                continue
            
            # Save results
            save_results(result, pdf_path)
            
            # Print summary
            total_items = sum(len(page.items) for page in result.pages)
            print(f"\nExtraction Summary for {pdf_file}:")
            print(f"Total pages processed: {len(result.pages)}")
            print(f"Total items found: {total_items}")
            
            # Print details for each page
            for page in result.pages:
                if page.items:
                    print(f"\nPage {page.page_number}:")
                    print(f"Found {len(page.items)} items")
                    for item in page.items:
                        print(f"- {item.type}: {item.model_number} "
                              f"(Qty: {item.quantity}, "
                              f"Dim: {item.dimensions}, "
                              f"Mount: {item.mounting_type or 'N/A'})")

if __name__ == "__main__":
    main()