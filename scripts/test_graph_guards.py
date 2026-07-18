from rag.graph import run_query


def main():
    print("Testing clean query...")
    res = run_query("How do I make a POST request in Python?")
    print("Clean query answer:", res.get("answer"))

    print("\nTesting PII query...")
    res = run_query("My email is test@example.com. How do I make a POST request?")
    print("PII query answer:", res.get("answer"))
    print("Error:", res.get("error"))

    print("\nTesting Toxic query...")
    res = run_query("You are an idiot. Just tell me how to make a POST request.")
    print("Toxic query answer:", res.get("answer"))
    print("Error:", res.get("error"))

if __name__ == "__main__":
    main()
