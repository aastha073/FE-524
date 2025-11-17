import os
import re
import json
import subprocess
from pathlib import Path
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define tools for the agent 
tools = [
    {
        "type": "function",
        "function":{
            "name": "determine_file_action",
            "description": "Determine what action to take for a given file based on its name and extension. Returns the action type: 'extract_and_parse_pdf', 'parse_txt', 'move_csv', 'summarize_doc', 'run_script', or 'invalid'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename to analyze, e.g., 'ibes_detail_history_docs_13.pdf'"
                    }
                },
                "required": ["filename"]
            }
        }    
    },
    {
        "type": "function",
        "function":{
            "name": "extract_pdf_text",
            "description": "Extract text content from a PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Full path to the PDF file"
                    }
                },
                "required": ["filepath"]
            }
          }  
    },
    {
        "type": "function",
        "function":{
            "name": "parse_ibes_to_csv",
            "description": "Parse IBES text data to CSV format and save to output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_content": {
                        "type": "string",
                        "description": "The IBES text content to parse"
                    },
                    "output_filename": {
                        "type": "string",
                        "description": "Output CSV filename (just the name, not full path)"
                    }
                },
                "required": ["text_content", "output_filename"]
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "create_summary",
            "description": "Create a brief summary or description of document page content and save to output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The page content to summarize"
                    },
                    "page_number": {
                        "type": "string",
                        "description": "The page number from the filename"
                    }
                },
                "required": ["content", "page_number"]
            }
         }
    },
    {
        "type": "function",
        "function":{
            "name": "execute_python_script",
            "description": "Execute a Python script file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Full path to the Python script"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "move_csv_to_output",
            "description": "Move a CSV file to the output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Source file path"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The filename"
                    }
                },
                "required": ["source_path", "filename"]
            }
        }
    }
]


# Tool implementations
def determine_file_action(filename):
    """Determine action based on filename patterns."""
    base = os.path.basename(filename)

    # IBES patterns
    if re.match(r'^ibes_(detail|summary)_history_docs_\d+\.pdf$', base, re.IGNORECASE):
        return json.dumps({"action": "extract_and_parse_pdf"})
    elif re.match(r'^ibes_(detail|summary)_history_docs_\d+\.txt$', base, re.IGNORECASE):
        return json.dumps({"action": "parse_txt"})
    elif re.match(r'^ibes_(detail|summary)_history_docs_\d+\.csv$', base, re.IGNORECASE):
        return json.dumps({"action": "move_csv"})

    # LSEG single-page patterns 
    elif re.match(r'^lseg_news_docs_\d+\.(pdf|txt)$', base, re.IGNORECASE):
        return json.dumps({"action": "summarize_doc"})

    # Python scripts
    elif base.lower().endswith('.py'):
        return json.dumps({"action": "run_script"})

    # Everything else is invalid
    else:
        return json.dumps({"action": "invalid"})

def extract_pdf_text(filepath):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return json.dumps({"success": True, "text": text})
    except ImportError:
        return json.dumps({"success": False, "error": "PyPDF2 not installed"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def parse_ibes_to_csv(text_content, output_filename):
    """Parse IBES text to CSV using LLM."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data parser. Convert IBES text data to CSV format. Return only the CSV content, no markdown formatting."},
                {"role": "user", "content": f"Parse this IBES data to CSV format:\n\n{text_content}"}
            ]
        )
        
        csv_content = response.choices[0].message.content
        output_path = Path("./output") / output_filename
        
        with open(output_path, 'w') as f:
            f.write(csv_content)
        
        return json.dumps({"success": True, "output_file": str(output_path)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def create_summary(content, page_number):
    """Create summary of page content using LLM."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document summarizer. Create brief, informative summaries of document pages."},
                {"role": "user", "content": f"Provide a brief summary or description of this document page:\n\n{content}"}
            ]
        )
        
        summary = response.choices[0].message.content
        output_filename = f"lseg_news_docs_{page_number}_summary.txt"
        output_path = Path("./output") / output_filename
        
        with open(output_path, 'w') as f:
            f.write(summary)
        
        return json.dumps({"success": True, "output_file": str(output_path)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def execute_python_script(filepath):
    """Execute a Python script."""
    try:
        result = subprocess.run(['python', filepath], capture_output=True, text=True, timeout=10)
        return json.dumps({
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def move_csv_to_output(source_path, filename):
    """Move CSV file to output directory."""
    try:
        import shutil
        destination = Path("./output") / filename
        shutil.copy2(source_path, destination)
        return json.dumps({"success": True, "destination": str(destination)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# Function dispatcher
available_functions = {
    "determine_file_action": determine_file_action,
    "extract_pdf_text": extract_pdf_text,
    "parse_ibes_to_csv": parse_ibes_to_csv,
    "create_summary": create_summary,
    "execute_python_script": execute_python_script,
    "move_csv_to_output": move_csv_to_output
}


def process_file_with_agent(filepath):
    """Process file using agent with tool calling pattern from class."""
    filename = os.path.basename(filepath)
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": "You are a file processing agent. You help determine what actions to take with different files and coordinate their processing. Use the available tools to process files appropriately."},
        {"role": "user", "content": f"I have a file named '{filename}' in the input directory. What should I do with it?"}
    ]
    
    print(f"\nProcessing: {filename}")
    
    # Agent reasoning loop
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Check if agent wants to use tools
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Agent calling: {function_name}")
                
                # Call the function
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                # Handle multi-step processing
                result = json.loads(function_response)
                
                # Special handling for determine_file_action
                if function_name == "determine_file_action":
                    action = result.get("action")
                    
                    if action == "invalid":
                        print(f"  Result: Invalid file, skipping")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": function_response
                        })
                        return
                    
                    # Continue processing based on action
                    if action == "extract_and_parse_pdf":
                        # Need to extract PDF, then parse
                        pdf_text_response = extract_pdf_text(str(filepath))
                        pdf_result = json.loads(pdf_text_response)
                        if pdf_result.get("success"):
                            output_name = filename.replace('.pdf', '.csv')
                            parse_ibes_to_csv(pdf_result["text"], output_name)
                            print(f"  Result: Extracted PDF and parsed to CSV")
                    
                    elif action == "parse_txt":
                        # Read txt file and parse
                        with open(filepath, 'r') as f:
                            content = f.read()
                        output_name = filename.replace('.txt', '.csv')
                        parse_ibes_to_csv(content, output_name)
                        print(f"  Result: Parsed TXT to CSV")
                    
                    elif action == "move_csv":
                        move_csv_to_output(str(filepath), filename)
                        print(f"  Result: Moved CSV to output")
                    
                    elif action == "summarize_doc":
                            # Read and summarize a single page (PDF or TXT)
                            if filename.lower().endswith(".pdf"):
                                pdf_text_response = extract_pdf_text(str(filepath))
                                pdf_result = json.loads(pdf_text_response)
                                if pdf_result.get("success"):
                                    content = pdf_result["text"]
                                else:
                                    print(f" Error extracting text: {pdf_result.get('error')}")
                                    return
                                
                            else:
                                # TXT
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()

                                # Extract page number for both PDF and TXT
                            m = re.search(r'_(\d+)\.(pdf|txt)$', filename, re.IGNORECASE)
                            if not m:
                                print(" Error: could not find page number in filename")
                                return
                            page_num = m.group(1)

                            # Create and save the summary
                            create_summary(content, page_num)
                            print(f" Result: Created summary")
                    
                    elif action == "run_script":
                        script_result = execute_python_script(str(filepath))
                        exec_result = json.loads(script_result)
                        if exec_result.get("success"):
                            print(f"  Result: Executed script")
                            if exec_result.get("stdout"):
                                print(f"  Output: {exec_result['stdout']}")
                
                # Add function response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_response
                })
        else:
            # No more tool calls, agent is done
            break
    
    return


def main():
    """Main agent loop."""
    input_dir = Path("./input")
    output_dir = Path("./output")
    
    # Create directories
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    print("File Processing Agent Started")
    print("=" * 60)
    
    # Get all files
    files = [f for f in input_dir.glob("*") if f.is_file()]
    
    if not files:
        print("No files found in ./input directory")
        return
    
    # Process each file with agent
    for filepath in files:
        try:
            process_file_with_agent(filepath)
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
    
    print("=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
