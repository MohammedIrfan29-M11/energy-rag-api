
import sys
sys.path.append('.')

from app.core.logging_config import setup_logging
from app.services.rag_service import query_rag

setup_logging()

print("\n" + "="*60)
print("ENERGY RAG SYSTEM — END TO END TEST")
print("="*60 + "\n")

# Test 1 — high confidence query
print("TEST 1: High confidence query")
print("-"*40)
result = query_rag("What are the renewable energy targets for 2030?")
print(f"Answer:\n{result['answer']}\n")
for i, citation in enumerate(result['citations']):
    print(f"  Citation {i+1}: {citation['source']} "
          f"(similarity: {citation['similarity']})")
print()

# Test 2 — conversation memory (follow up question)
print("TEST 2: Follow-up question (tests conversation memory)")
print("-"*40)
followup = query_rag(
    "Which countries are leading in achieving these targets?",
    history=result['history']  # pass history from test 1
)
print(f"Answer:\n{followup['answer']}\n")
print()

# Test 3 — below threshold query
print("TEST 3: Out of scope query (should refuse)")
print("-"*40)
out_of_scope = query_rag("What is the best recipe for hummus?")
print(f"Answer:\n{out_of_scope['answer']}\n")
print()

# Test 4 — complex query
print("TEST 4: Complex energy policy query")
print("-"*40)
complex_result = query_rag(
    "What is the IEA's projection for fossil fuel demand "
    "beyond 2030 and what are the main drivers?"
)
print(f"Answer:\n{complex_result['answer']}\n")
print(f"Citations: {len(complex_result['citations'])}")