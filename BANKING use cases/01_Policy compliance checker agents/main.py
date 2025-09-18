import os
import logging
from dotenv import load_dotenv
from workflows.compliance_workflow import ComplianceWorkflow
from rich.console import Console
import sys

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compliance_workflow.log'),
        logging.StreamHandler()
    ]
)

console = Console()

def main():
    """Main function to run the compliance workflow."""
    try:
        # Get API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            console.print("[bold red]Error: GOOGLE_API_KEY not found in environment variables[/bold red]")
            return
        
        # Get input file paths
        console.print("[bold blue]Bank Policy Compliance Automation System[/bold blue]")
        console.print("\nPlease ensure your documents are in the 'input/' folder")
        
        bank_policy_path = input("Enter bank policy file name (in input/ folder): ")
        regulatory_doc_path = input("Enter regulatory document file name (in input/ folder): ")
        
        # Prepend input folder path
        bank_policy_path = os.path.join("input", bank_policy_path)
        regulatory_doc_path = os.path.join("input", regulatory_doc_path)
        
        # Validate files exist
        if not os.path.exists(bank_policy_path):
            console.print(f"[bold red]Error: Bank policy file not found: {bank_policy_path}[/bold red]")
            return
        
        if not os.path.exists(regulatory_doc_path):
            console.print(f"[bold red]Error: Regulatory document not found: {regulatory_doc_path}[/bold red]")
            return
        
        # Initialize workflow
        workflow = ComplianceWorkflow(api_key, output_dir="output")
        
        # Run workflow
        console.print("\n[bold yellow]Starting compliance analysis...[/bold yellow]")
        result = workflow.run_workflow(bank_policy_path, regulatory_doc_path)
        
        # Display summary
        console.print("\n[bold green]🎉 Compliance Analysis Complete![/bold green]")
        console.print(f"📊 Total findings: {len(result.get('findings', []))}")
        console.print(f"📁 Reports saved in: {workflow.output_dir}")
        console.print(f"📋 Check the following files:")
        console.print(f"   - compliance_report.docx")
        console.print(f"   - compliance_report.json")
        console.print(f"   - Individual requirement files (*.json)")
        
        # Display compliance summary
        if result.get('findings'):
            console.print("\n[bold blue]Compliance Summary:[/bold blue]")
            status_counts = {}
            for finding in result['findings']:
                status = finding.get('compliance_status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            for status, count in status_counts.items():
                console.print(f"   {status.replace('_', ' ').title()}: {count}")
        
        console.print("\n[bold cyan]Analysis complete! Check the output folder for detailed reports.[/bold cyan]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logging.error(f"Main execution error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
