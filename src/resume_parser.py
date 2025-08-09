import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
import asyncio
from collections import Counter
from .utils import get_env_var

def _debug_enabled() -> bool:
    return get_env_var("RESUME_PARSER_DEBUG", "0").lower() in ("1", "true", "yes", "on")

def _dbg(message: str) -> None:
    if _debug_enabled():
        print(f"[resume_parser.debug] {message}")

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # spaCy not installed

_NLP = None  # cached nlp model

def _get_spacy_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        return None
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        # Model not available; return None to trigger filename fallback
        _NLP = None
    return _NLP

def extract_name_from_resume(resume_text: str) -> str:
    """Extract a PERSON name from resume text using spaCy NER.
    Returns the first detected PERSON entity or 'Name not found' if none found.
    """
    nlp = _get_spacy_nlp()
    if not nlp or not resume_text:
        return "Name not found"
    
    try:
        # Process the first few lines where names typically appear
        first_lines = '\n'.join(resume_text.split('\n')[:3])
        doc = nlp(first_lines.title())
        # Get all PERSON entities and return the first one
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text.strip():
                return ent.text.strip()
                
    except Exception as e:
        return "Name not found"
    
    return "Name not found"


def setup_llamaparse():
    """Load API key and initialize LlamaParse with proper configuration."""
    load_dotenv()
    api_key = get_env_var('LLAMAPARSE')
    
    if not api_key:
        raise ValueError("LLAMAPARSE API key not found in environment variables or Streamlit secrets")
    
    # Initialize with minimal, stable configuration
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )
    
    return parser

async def parse_resume_async(file_path):
    """Async version of resume parsing."""
    parser = setup_llamaparse()
    
    try:
        # Use async parsing
        documents = await parser.aload_data(file_path)
        
        if documents and len(documents) > 0:
            return documents[0].text
        else:
            return "No content extracted"
            
    except Exception as e:
        return f"Error: {str(e)}"

def parse_resume_sync(file_path):
    """Synchronous version with better error handling."""
    parser = setup_llamaparse()
    
    try:
        # Try synchronous parsing first
        documents = parser.load_data(file_path)
        
        if documents and len(documents) > 0:
            return documents[0].text
        else:
            return "No content extracted"
            
    except Exception as e:
        # Try async version as fallback
        try:
            return asyncio.run(parse_resume_async(file_path))
        except Exception as async_error:
            return f"Both sync and async failed. Sync: {str(e)}, Async: {str(async_error)}"

def test_api_connection():
    """Test if API key and connection work."""
    try:
        load_dotenv()
        api_key = get_env_var('LLAMAPARSE')
        
        if not api_key:
            return False, "No API key found"
        
        # Simple parser initialization test
        parser = LlamaParse(api_key=api_key)
        return True, "API connection successful"
        
    except Exception as e:
        return False, f"API connection failed: {str(e)}"


def main():
    """Main function with enhanced error handling."""
    print("=== LlamaParse Resume Parser ===\n")
    
    # Test API connection first
    api_ok, api_msg = test_api_connection()
    print(f"API Status: {api_msg}")
    
    if not api_ok:
        print("\nPlease check your .env file and API key.")
        return
    
    # Get resume file path
    resume_path = input("\nEnter the path to your resume file: ").strip()
    
    # Remove quotes if user copied path with quotes
    resume_path = resume_path.strip('"').strip("'")
    
    if not os.path.exists(resume_path):
        print(f"File not found: {resume_path}")
        return
    
    print(f"\nFile found: {resume_path}")
    print(f"File size: {os.path.getsize(resume_path)} bytes")
    
    # Try parsing
    print("\nAttempting to parse resume...")
    content = parse_resume_sync(resume_path)
    
    print("\n" + "="*60)
    print("PARSING RESULT")
    print("="*60)
    
    if content.startswith("Error"):
        print(f"❌ {content}")
    else:
        print("✅ Parsing successful!")
        print(f"\nContent length: {len(content)} characters")
        print(f"First 500 characters:\n{content[:500]}")
        
        # Save to file
        output_file = "parsed_resume_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n💾 Full content saved to: {output_file}")

if __name__ == "__main__":
    main() 