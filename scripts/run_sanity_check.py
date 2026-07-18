import os
import sys
import time

# Add the src directory to the sys path so we can import rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rag.graph import run_query

questions = [
    "How do I send a basic GET request?",
    "How do I pass parameters in URLs?",
    "How do I read the response content as JSON?",
    "How do I send a POST request with form data?",
    "How do I set custom headers on my request?",
    "How do I check the response status code?",
    "How do I handle cookies in a request?",
    "How do I set a timeout on a request?",
    "What exception is raised if a request times out?",
    "How do I use sessions for connection pooling?",
    "How do I perform basic HTTP authentication?",
    "How do I upload a file in a POST request?",
    "How do I verify SSL certificates?",
    "How do I access the underlying urllib3 response?",
    "How do I define a custom proxy?"
]

def main():
    print("Running Sanity Check against 15 test queries...")

    results = []
    total_latency = 0.0

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/15] Question: {q}")
        start_t = time.perf_counter()

        try:
            res = run_query(q, verbose=False)
        except Exception as e:
            print(f"Error running query: {e}")
            res = {"answer": f"ERROR: {str(e)}", "latency_ms": 0.0, "sources_used": []}

        end_t = time.perf_counter()
        query_latency = res.get("latency_ms", (end_t - start_t) * 1000)

        total_latency += query_latency

        answer = res.get("answer", "(No answer)")
        sources = res.get("sources_used", [])

        print(f"Latency: {query_latency:.2f} ms")
        print(f"Sources: {sources}")
        print(f"Answer snippet: {answer[:100]}...")

        results.append({
            "question": q,
            "answer": answer,
            "latency_ms": query_latency,
            "sources": sources
        })

    avg_latency = total_latency / len(questions)
    print("\n--- DONE ---")
    print(f"Average Latency: {avg_latency:.2f} ms")

    # Generate Markdown
    md_content = "# Baseline Metrics\n\n"
    md_content += "This document tracks the baseline latency and sanity check results for the linear RAG pipeline (v1).\n\n"
    md_content += f"**Test Run Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"**Total Queries:** {len(questions)}\n"
    md_content += f"**Average Latency:** {avg_latency:.2f} ms\n\n"
    md_content += "## Query Results\n\n"

    for i, r in enumerate(results, 1):
        md_content += f"### {i}. {r['question']}\n"
        md_content += f"- **Latency:** {r['latency_ms']:.2f} ms\n"
        md_content += f"- **Sources Used:** {r['sources']}\n\n"
        md_content += f"**Answer:**\n\n> {r['answer']}\n\n"
        md_content += "---\n\n"

    # Write to docs/baseline_metrics.md
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs'))
    os.makedirs(docs_dir, exist_ok=True)

    metrics_file = os.path.join(docs_dir, 'baseline_metrics.md')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Saved results to {metrics_file}")

if __name__ == '__main__':
    main()
