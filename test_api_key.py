#!/usr/bin/env python3
"""Test if .env file is loading correctly"""

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print("✓ API key loaded successfully!")
    print(f"  Key starts with: {api_key[:20]}...")
    print(f"  Key ends with: ...{api_key[-10:]}")
    print(f"  Key length: {len(api_key)} characters")
    
    # Check for common issues
    if api_key.startswith('"') or api_key.endswith('"'):
        print("\n⚠️  WARNING: API key has quotes around it in .env file")
        print("   Remove quotes from .env file (should be: OPENAI_API_KEY=sk-proj-...)")
    
    if ' ' in api_key:
        print("\n⚠️  WARNING: API key contains spaces")
    
    if not api_key.startswith('sk-'):
        print("\n⚠️  WARNING: API key doesn't start with 'sk-'")
    
    # Test the API key with OpenAI
    print("\nTesting API key with OpenAI...")
    try:
        from openai import OpenAI
        client = OpenAI()
        # Make a simple test call
        response = client.models.list()
        print("✓ API key is VALID! OpenAI connection successful.")
    except Exception as e:
        print(f"✗ API key is INVALID: {e}")
        print("\nPossible reasons:")
        print("  1. The API key is expired or revoked")
        print("  2. You need to create a new key at: https://platform.openai.com/api-keys")
        print("  3. Your OpenAI account needs billing/credits enabled")
else:
    print("✗ API key NOT loaded from .env file")
    print("\nCheck that:")
    print("  1. .env file exists in the current directory")
    print("  2. .env file contains: OPENAI_API_KEY=sk-proj-...")
    print("  3. No quotes around the key")
    print("  4. No spaces around the = sign")




