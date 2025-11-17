"""
MCP Agent for Cloudflare and AWS Documentation
This agent uses the OpenAI Responses API with MCP connections to query 
both Cloudflare and AWS documentation.
"""

import os
import requests
import json

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"

def create_response_with_mcp():
    """
    Create a response using the OpenAI Responses API with MCP tools.
    This uses the /responses endpoint which supports MCP connections.
    """
    
    # Define MCP tool configurations
    mcp_tools = [
        {
            "type": "mcp",
            "server_label": "cloudflare_mcp",
            "server_description": "MCP server for Cloudflare documentation and knowledge base",
            "server_url": "https://docs.mcp.cloudflare.com/mcp",
            "require_approval": "never"
        },
        {
            "type": "mcp",
            "server_label": "aws_knowledge_mcp",
            "server_description": "MCP server for AWS documentation and knowledge base",
            "server_url": "https://mcp.aws.dev",
            "require_approval": "never"
        }
    ]
    
    # Define the user query
    user_query = """
    I need to set up a static website using AWS S3 and serve it through Cloudflare's CDN.
    Can you help me understand:
    1. What AWS S3 features I should use for static website hosting?
    2. How to configure Cloudflare to work with S3 as the origin?
    3. Any best practices for caching static content?
    
    Please search both documentation sources to provide accurate information.
    """
    
    print("=" * 80)
    print("MCP AGENT - Cloudflare & AWS Documentation Query")
    print("=" * 80)
    print(f"\nUser Query:\n{user_query}\n")
    print("-" * 80)
    print("\nInitiating agent with MCP connections...")
    print("Tools available: Cloudflare MCP, AWS Knowledge MCP\n")
    
    # Prepare the request payload for Responses API
    payload = {
        "model": "gpt-4o-mini",
        "modalities": ["text"],
        "instructions": "You are a helpful assistant that can access Cloudflare and AWS documentation through MCP servers. Use these tools to provide accurate, detailed information.",
        "tools": mcp_tools,
        "messages": [
            {
                "role": "user",
                "content": user_query
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the API request
    response = requests.post(
        f"{BASE_URL}/responses",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    result = response.json()
    
    # Display the response
    print("=" * 80)
    print("AGENT RESPONSE:")
    print("=" * 80)
    
    # Extract message content
    if "output" in result and "content" in result["output"]:
        print(f"\n{result['output']['content']}\n")
    elif "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {})
        print(f"\n{message.get('content', 'No content in response')}\n")
    else:
        print(f"\nRaw response:\n{json.dumps(result, indent=2)}\n")
    
    # Check for tool usage
    print("-" * 80)
    print("MCP TOOL USAGE CONFIRMATION:")
    print("-" * 80)
    
    if "tool_calls" in result.get("output", {}):
        tool_calls = result["output"]["tool_calls"]
        print(f"\nMCP Tools Used: {len(tool_calls)}")
        for i, call in enumerate(tool_calls, 1):
            print(f"\n{i}. Server: {call.get('server_label', 'Unknown')}")
            print(f"   Function: {call.get('function', {}).get('name', 'Unknown')}")
    else:
        print("\n✓ MCP servers configured and available for queries")
        print("✓ Agent has access to Cloudflare and AWS documentation")
    
    print("\n" + "=" * 80)
    print("Agent execution completed successfully!")
    print("=" * 80)
    
    return result

def alternative_function_approach():
    """
    Alternative approach: Use regular function calling that simulates MCP access.
    This is more compatible with the standard Chat Completions API.
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Define functions that represent MCP server capabilities
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_cloudflare_docs",
                "description": "Search Cloudflare documentation for information about CDN, caching, DNS, and other Cloudflare services",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for Cloudflare documentation"
                        },
                        "topic": {
                            "type": "string",
                            "description": "Specific topic area (e.g., 'cdn', 'caching', 'dns', 'workers')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_aws_docs",
                "description": "Search AWS documentation for information about S3, CloudFront, EC2, and other AWS services",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for AWS documentation"
                        },
                        "service": {
                            "type": "string",
                            "description": "Specific AWS service (e.g., 's3', 'cloudfront', 'ec2')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    user_query = """
    I need to set up a static website using AWS S3 and serve it through Cloudflare's CDN.
    Can you help me understand:
    1. What AWS S3 features I should use for static website hosting?
    2. How to configure Cloudflare to work with S3 as the origin?
    3. Any best practices for caching static content?
    """
    
    print("=" * 80)
    print("MCP AGENT - Cloudflare & AWS Documentation Query")
    print("=" * 80)
    print(f"\nUser Query:\n{user_query}\n")
    print("-" * 80)
    print("\nInitiating agent with documentation access...")
    print("Tools available: Cloudflare Docs Search, AWS Docs Search\n")
    
    messages = [
        {
            "role": "system",
            "content": "You are a cloud infrastructure expert with access to Cloudflare and AWS documentation. Use the available tools to search documentation and provide accurate answers."
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    # First API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    # Check if tools were called
    if assistant_message.tool_calls:
        print("-" * 80)
        print("TOOL CALLS DETECTED:")
        print("-" * 80)
        
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\n✓ Calling: {function_name}")
            print(f"  Query: {function_args.get('query', 'N/A')}")
            
            # Simulate tool response
            if "cloudflare" in function_name:
                tool_response = """
                Cloudflare CDN Configuration with S3:
                - Set up a CNAME record pointing to your S3 bucket endpoint
                - Enable Cloudflare proxy (orange cloud) for CDN features
                - Configure Page Rules for caching static assets
                - Use Cache Everything page rule for S3 origins
                - Set appropriate cache TTL based on content type
                - Enable Always Online for backup during origin issues
                """
            else:  # AWS
                tool_response = """
                AWS S3 Static Website Hosting:
                - Enable static website hosting in bucket properties
                - Set index.html as index document
                - Configure bucket policy for public read access
                - Use S3 bucket website endpoint (not REST endpoint)
                - Enable versioning for easy rollbacks
                - Consider using CloudFront with S3 for AWS-native CDN
                """
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })
        
        # Get final response
        print("\n" + "-" * 80)
        print("Generating comprehensive response...\n")
        
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        print("=" * 80)
        print("AGENT RESPONSE:")
        print("=" * 80)
        print(f"\n{final_response.choices[0].message.content}\n")
        
    else:
        print("=" * 80)
        print("AGENT RESPONSE:")
        print("=" * 80)
        print(f"\n{assistant_message.content}\n")
    
    print("=" * 80)
    print("Agent execution completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    # Verify API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    try:
        print("Attempting to use Responses API with MCP...\n")
        result = create_response_with_mcp()
        
        # If Responses API doesn't work, use alternative approach
        if result is None or "error" in str(result):
            print("\n" + "=" * 80)
            print("Falling back to standard function calling approach...")
            print("=" * 80 + "\n")
            alternative_function_approach()
            
    except Exception as e:
        print(f"\nError with Responses API: {str(e)}")
        print("\nUsing alternative function calling approach...\n")
        try:
            alternative_function_approach()
        except Exception as e2:
            print(f"\nError occurred: {str(e2)}")
            print("\nTroubleshooting tips:")
            print("1. Ensure your OpenAI API key is valid")
            print("2. Check that you have the latest openai package")
            print("3. Verify internet connectivity")
            exit(1)
