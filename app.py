from qa_assistant import answer_question

print("📄 Contract Clause Q&A Assistant")
print("Ask a question like: 'Does this agreement restrict assignment?'")

while True:
    q = input("\nYour question: ")
    if q.lower() in ["quit","exit"]:
        break
    print("\n--- Answer ---\n")
    print(answer_question(q))
