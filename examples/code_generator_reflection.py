"""
Example: Self-Reflecting Code Generator.

This uses a ReflectionAgent to draft, critique, and refine a python script
before returning it to the user.
"""

import asyncio
from agentkit import Agent
from agentkit.orchestration.reflection import ReflectionAgent

async def main():
    print("Initializing ReflectionAgent...")
    
    # We use a standard Agent internally
    base_agent = Agent("CodeGenerator", model="gpt-4o-mini", temperature=0.7)
    
    # Wrap it in the Reflection loop
    reflect_agent = ReflectionAgent(
        agent=base_agent,
        max_reflections=3,
        rubric="""
        1. Does the code contain syntax errors?
        2. Are there appropriate docstrings and comments?
        3. Are variables named clearly?
        4. Does it handle edge cases?
        5. Return ONLY python code in a markdown block, no extra conversational text.
        """
    )
    
    prompt = "Write a python function to compute the Fibonacci sequence up to n numbers using a generator, then print the first 10."
    print(f"\nPrompt: {prompt}\n")
    print("Drafting and Refining (this will loop a few times)...\n")
    
    try:
        # arun will handle the drafting, critiquing, and refining automatically
        final_code = await reflect_agent.arun(prompt)
        
        print("\n" + "="*50)
        print("FINAL REFINED CODE:")
        print("="*50)
        print(final_result)
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
