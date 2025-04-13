from typing import Dict, Any, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph
import gc
import torch

from core.chains.extraction import PDFExtractionChain
from core.chains.validation import ValidationChain
from core.models.plumbing import ExtractionResult

class WorkflowState(TypedDict):
    """State for the plumbing data extraction workflow."""
    pdf_path: str
    extraction_result: ExtractionResult | None
    validation_result: ExtractionResult | None
    error: str | None
    retry_count: int

def create_plumbing_workflow(model_name: str = "llama3") -> Graph:
    """Create the plumbing data extraction workflow."""
    
    # Initialize chains
    extraction_chain = PDFExtractionChain(model_name=model_name)
    validation_chain = ValidationChain(model_name=model_name)
    
    # Define workflow nodes
    def extract_data(state: WorkflowState) -> WorkflowState:
        """Extract data from PDF."""
        try:
            # Clear memory before extraction
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = extraction_chain.extract_from_pdf(state["pdf_path"])
            return {"extraction_result": result, "error": None, "retry_count": 0}
        except Exception as e:
            return {"error": str(e), "retry_count": state.get("retry_count", 0) + 1}
    
    def validate_data(state: WorkflowState) -> WorkflowState:
        """Validate extracted data."""
        try:
            if state["extraction_result"] is None:
                return {"error": "No extraction result to validate", "retry_count": state.get("retry_count", 0)}
            
            # Clear memory before validation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = validation_chain.validate_extraction_result(state["extraction_result"])
            return {"validation_result": result, "error": None, "retry_count": 0}
        except Exception as e:
            return {"error": str(e), "retry_count": state.get("retry_count", 0)}
    
    def handle_error(state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        if state["error"]:
            print(f"Error in workflow: {state['error']}")
            
            # If we've retried too many times, stop
            if state.get("retry_count", 0) >= 3:
                print("Max retries reached. Stopping workflow.")
                return state
            
            # Clear memory before retry
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # If we have an extraction result but validation failed, return that
            if state.get("extraction_result") and not state.get("validation_result"):
                print("Returning extraction result after validation failure")
                return state
            
            # Otherwise, try extraction again
            print("Retrying extraction...")
            return {"pdf_path": state["pdf_path"], "retry_count": state.get("retry_count", 0)}
        
        return state
    
    # Create workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("extract", extract_data)
    workflow.add_node("validate", validate_data)
    workflow.add_node("error_handler", handle_error)
    
    # Add edges
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "error_handler")
    
    # Only retry extraction if we haven't exceeded retry limit
    def should_retry(state: WorkflowState) -> bool:
        return state.get("retry_count", 0) < 3
    
    workflow.add_conditional_edges(
        "error_handler",
        {
            "extract": should_retry,
            "end": lambda x: not should_retry(x)
        }
    )
    
    # Set entry point
    workflow.set_entry_point("extract")
    
    # Compile workflow
    return workflow.compile()

def run_workflow(pdf_path: str, model_name: str = "llama3") -> ExtractionResult:
    """Run the plumbing data extraction workflow."""
    # Create workflow
    workflow = create_plumbing_workflow(model_name)
    
    # Initialize state
    initial_state = WorkflowState(
        pdf_path=pdf_path,
        extraction_result=None,
        validation_result=None,
        error=None,
        retry_count=0
    )
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    # Return validation result if successful
    if final_state["validation_result"]:
        return final_state["validation_result"]
    
    # Return extraction result if validation failed
    if final_state["extraction_result"]:
        return final_state["extraction_result"]
    
    # Raise error if both failed
    raise Exception(final_state["error"] or "Workflow failed") 