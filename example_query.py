from rag_retriever import retrieve

print("Ask a question about the contract, e.g.:")
print(" → 'Can the contract be terminated without cause?'")

while True:
    q = input("\nYour question: ")
    if q.lower() in ("quit","exit"): break

    results = retrieve(q)
    print("\nTop Clauses:\n")
    for i, clause in range(len(results)):
        print(f"--- Clause {i+1} ---\n{results[i]}\n")
