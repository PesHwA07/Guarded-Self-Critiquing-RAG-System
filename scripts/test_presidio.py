import logging
logging.basicConfig(level=logging.INFO)

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def main():
    try:
        analyzer = AnalyzerEngine()
        print("Analyzer instantiated successfully.")
        
        results = analyzer.analyze(text="My phone number is 212-555-5555", language='en')
        print(f"Analysis results: {results}")
    except Exception as e:
        print(f"Failed to instantiate Analyzer: {e}")

if __name__ == "__main__":
    main()
