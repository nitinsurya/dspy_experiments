# Ensure you have the necessary libraries installed:
# pip install dspy-ai litellm openai PyYAML # openai is needed by dspy, PyYAML can be helpful
# Example: pip install dspy-ai==2.4.3 litellm==1.37.10 openai==1.28.1 PyYAML==6.0.1

import dspy
import litellm
import os
import json
import logging
import ast # For safely evaluating string representations of lists
from typing import List, Optional, Dict, Any

# --- Configuration ---

# IMPORTANT: Make sure your Ollama server is running and has the model pulled:
# ollama pull mistral-small3.1
# ollama serve (or ensure it's running in the background)

# Specify the Ollama model to use via litellm
# The format "ollama/model_name" tells litellm to use the Ollama provider
# Make sure the base_url points to your running Ollama instance
OLLAMA_MODEL_NAME = "ollama/mistral-small3.1"
DSPY_OLLAMA_MODEL_NAME = "ollama_chat/mistral-small3.1"
OLLAMA_API_BASE = "http://localhost:11434" # Default Ollama API base

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Approach 1: Simple Prompting with litellm ---

def analyze_sentiment_simple(review: str) -> Dict[str, Any]:
    """
    Analyzes sentiment using a direct prompt with litellm.

    Args:
        review: The movie review text.

    Returns:
        A dictionary containing sentiment, explanation, and keywords,
        or an error dictionary if parsing fails or API call fails.
    """
    logger.info("\n--- Running Simple Prompting ---")
    # Define the prompt using an f-string. Ensure the closing triple quotes are correct.
    prompt = f"""\
Analyze the sentiment of the following movie review. Determine if it is positive, negative, or neutral.
If the sentiment is positive or negative, provide a brief explanation (1-2 sentences).
Extract the main keywords related to the sentiment and the movie's aspects mentioned.

Respond ONLY with a valid JSON object containing the following keys:
- "sentiment": A string, must be one of "positive", "negative", or "neutral".
- "explanation": A string containing the explanation, or null if the sentiment is neutral.
- "keywords": A JSON list of strings.

Movie Review:
"{review}"

JSON Response:
""" # End of the f-string. The comment is correctly placed *after* the closing quotes.

    try:
        logger.info("Sending request to Ollama via litellm...")
        # Uncomment the line below for verbose litellm logs during debugging
        # litellm.set_verbose = True
        response = litellm.completion(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            api_base=OLLAMA_API_BASE,
            temperature=0.1, # Lower temperature for more deterministic output
            max_tokens=200,  # Increased slightly for JSON structure + explanation
            request_timeout=60 # Add a timeout to prevent hanging indefinitely
        )
        # Disable verbose logs after the call if they were enabled
        # litellm.set_verbose = False

        logger.info("Received response from Ollama.")
        # Extract the response content
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             logger.error("Error: Received empty or invalid response structure from litellm.")
             return {"error": "Received empty response from LLM"}

        content = response.choices[0].message.content.strip()
        logger.info(f"Raw response content:\n{content}")

        # Attempt to parse the JSON response
        result = None # Initialize result to None
        try:
            # Clean potential markdown code blocks (` ```json ... ``` `) and extraneous text
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = content[json_start:json_end+1]
                logger.info(f"Extracted JSON string: {json_string}")
                result = json.loads(json_string)
            else:
                 # Fallback if no valid braces found
                 logger.error("Could not find valid JSON object delimiters {{...}} in the response.")
                 raise json.JSONDecodeError("No JSON object found", content, 0)

            # --- Validation ---
            required_keys = ["sentiment", "explanation", "keywords"]
            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing required keys in JSON response: {missing_keys}")

            valid_sentiments = ["positive", "negative", "neutral"]
            if result.get("sentiment") not in valid_sentiments:
                raise ValueError(f"Invalid sentiment value: '{result.get('sentiment')}'")

            if not isinstance(result.get("keywords"), list):
                 raise ValueError(f"Keywords should be a list, got: {type(result.get('keywords'))}")

            # Ensure explanation logic is correct
            if result["sentiment"] == "neutral" and result.get("explanation") is not None:
                 logger.warning("Explanation provided for neutral sentiment, setting to None.")
                 result["explanation"] = None
            # Ensure explanation is present for non-neutral, allow empty string but log warning if null/None
            elif result["sentiment"] != "neutral" and result.get("explanation") is None:
                 logger.warning("Explanation missing (is null) for non-neutral sentiment. Setting to empty string.")
                 result["explanation"] = "" # Default to empty string if required

            logger.info("Successfully parsed and validated JSON response.")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Error: Failed to parse JSON response: {e}")
            logger.error(f"Content attempted to parse: {json_string if 'json_string' in locals() else content}")
            return {"error": "Failed to parse LLM response as JSON", "raw_content": content}
        except ValueError as e:
            logger.error(f"Error: Invalid JSON structure or values: {e}")
            # Log the partially parsed content if available
            logger.error(f"Parsed content (if available) before error: {result}")
            return {"error": f"Invalid JSON structure or value: {e}", "parsed_content_before_error": result}

    except litellm.exceptions.APIConnectionError as e:
         logger.error(f"Error: Connection to Ollama API failed at {OLLAMA_API_BASE}: {e}", exc_info=True)
         return {"error": f"API Connection Error: {e}"}
    except litellm.exceptions.Timeout as e:
         logger.error(f"Error: Request to Ollama timed out: {e}", exc_info=True)
         return {"error": f"API Timeout Error: {e}"}
    except Exception as e:
        # Catch other potential litellm errors or general exceptions
        logger.error(f"Error during litellm completion call: {e}", exc_info=True)
        return {"error": f"LLM API call failed unexpectedly: {e}"}

# --- Approach 2: DSPy ---

# 1. Define the Signature for the task
class SentimentAnalysisSignature(dspy.Signature):
    """Analyzes movie review sentiment, providing explanation and keywords.
    Input: A movie review.
    Output: Sentiment (positive/negative/neutral), explanation (if applicable), and keywords (as a list)."""

    movie_review = dspy.InputField(desc="The text of the movie review.")
    sentiment = dspy.OutputField(desc="The overall sentiment (must be one of 'positive', 'negative', or 'neutral').")
    explanation = dspy.OutputField(desc="A brief explanation for positive/negative sentiment (should be null or empty string if sentiment is neutral).")
    # Specify output format clearly for keywords
    keywords = dspy.OutputField(desc="A Python-style list of strings (e.g., ['keyword1', 'keyword2']) representing keywords related to the sentiment and movie aspects.")

# 2. Configure the Language Model
# Use dspy.Ollama which internally uses litellm
# Pass model name, api_base, and any other litellm parameters
llm = None # Initialize llm to None
# Extract only the model name part for dspy.Ollama constructor
ollama_model_id = OLLAMA_MODEL_NAME.split('/')[-1] if '/' in OLLAMA_MODEL_NAME else OLLAMA_MODEL_NAME

llm = dspy.LM(
    model=DSPY_OLLAMA_MODEL_NAME,
    api_base=OLLAMA_API_BASE,
    max_tokens=200, # Consistent with simple approach
    temperature=0.1,
    request_timeout=60 # Add timeout
)
# Configure DSPy settings to use this LLM
dspy.settings.configure(lm=llm)
logger.info(f"DSPy configured with Ollama model '{ollama_model_id}' at {OLLAMA_API_BASE}.")

# 3. Create the Predictor
# Check if llm was configured successfully
sentiment_predictor = None # Initialize predictor to None
if llm:
    try:
        sentiment_predictor = dspy.Predict(SentimentAnalysisSignature)
        logger.info("DSPy Predict module created successfully.")
    except Exception as e:
        logger.error(f"Failed to create DSPy Predict module: {e}", exc_info=True)
else:
    logger.error("DSPy Predict module cannot be created because LM configuration failed.")


def analyze_sentiment_dspy(review: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes sentiment using a DSPy Predict module.

    Args:
        review: The movie review text.

    Returns:
        A dictionary containing processed sentiment, explanation, and keywords,
        or None if DSPy setup failed or prediction encountered an error.
    """
    if not sentiment_predictor:
        logger.error("DSPy predictor is not available. Cannot analyze sentiment.")
        return None # Return None if predictor wasn't created

    logger.info("\n--- Running DSPy Predict ---")
    logger.info("Sending request via DSPy...")

    try:
        # DSPy handles the prompting and parsing based on the signature
        # It might retry or handle errors internally depending on configuration
        prediction = sentiment_predictor(movie_review=review)
        logger.info("Received prediction object from DSPy.")
        # Log the raw prediction fields for debugging
        logger.debug(f"Raw DSPy prediction sentiment: {getattr(prediction, 'sentiment', 'MISSING')}")
        logger.debug(f"Raw DSPy prediction explanation: {getattr(prediction, 'explanation', 'MISSING')}")
        logger.debug(f"Raw DSPy prediction keywords: {getattr(prediction, 'keywords', 'MISSING')}")


        # --- Validation and Cleanup ---
        # DSPy attempts structured output, but validation is still essential.

        processed_sentiment = getattr(prediction, 'sentiment', None)
        processed_explanation = getattr(prediction, 'explanation', None)
        raw_keywords = getattr(prediction, 'keywords', None)
        parsed_keywords = [] # Default to empty list

        # Validate sentiment
        valid_sentiments = ["positive", "negative", "neutral"]
        if processed_sentiment not in valid_sentiments:
             logger.warning(f"Received unexpected or missing sentiment: '{processed_sentiment}'. Setting to None.")
             processed_sentiment = None # Or handle as error depending on requirements

        # Validate explanation based on (validated) sentiment
        if processed_sentiment == "neutral":
            # If sentiment is neutral, explanation should ideally be None or empty
            if processed_explanation and processed_explanation.strip().lower() not in ['null', 'none', '']:
                logger.warning(f"Explanation '{processed_explanation}' provided for neutral sentiment. Clearing explanation.")
                processed_explanation = None # Enforce None for neutral
        elif processed_sentiment in ["positive", "negative"]:
             # If sentiment is non-neutral, explanation should ideally exist
            if not processed_explanation or processed_explanation.strip().lower() in ['null', 'none', '']:
                logger.warning(f"Explanation missing or null/empty for non-neutral sentiment '{processed_sentiment}'. Setting to empty string.")
                processed_explanation = "" # Provide a default empty string

        # Validate and parse keywords (DSPy might return a string representation of a list)
        if isinstance(raw_keywords, list):
            # Check if elements are strings
            if all(isinstance(item, str) for item in raw_keywords):
                 parsed_keywords = raw_keywords # Already a valid list of strings
            else:
                 logger.warning(f"Keywords field is a list, but contains non-string elements: {raw_keywords}. Attempting to convert.")
                 parsed_keywords = [str(item) for item in raw_keywords]

        elif isinstance(raw_keywords, str):
            kw_str = raw_keywords.strip()
            logger.info(f"Attempting to parse keywords string: '{kw_str}'")
            if kw_str: # Only parse if not empty
                try:
                    # Use ast.literal_eval for safe evaluation of Python literals (like lists)
                    evaluated = ast.literal_eval(kw_str)
                    if isinstance(evaluated, list):
                         # Further check if list items are strings
                         if all(isinstance(item, str) for item in evaluated):
                              parsed_keywords = evaluated
                              logger.info(f"Successfully parsed keywords string using ast: {parsed_keywords}")
                         else:
                              logger.warning(f"Parsed keyword string with ast, but result contains non-strings: {evaluated}. Converting.")
                              parsed_keywords = [str(item) for item in evaluated]
                    else:
                         logger.warning(f"Parsed keyword string with ast, but result is not a list: {evaluated}. Type: {type(evaluated)}. Treating as single keyword.")
                         parsed_keywords = [kw_str] # Fallback: treat original string as single keyword

                except (ValueError, SyntaxError, MemoryError) as e:
                    # ast.literal_eval failed, likely not a valid Python literal string
                    logger.warning(f"Could not parse keywords string '{kw_str}' using ast.literal_eval: {e}. Splitting by comma as fallback.")
                    # Fallback: split by comma, removing empty strings
                    parsed_keywords = [k.strip() for k in kw_str.split(',') if k.strip()]
                    if not parsed_keywords: # If split results in empty list, use original string
                         parsed_keywords = [kw_str]
            else:
                logger.info("Keywords string is empty.")
                parsed_keywords = []

        else:
            logger.warning(f"Keywords field has unexpected type: {type(raw_keywords)}. Expected list or string. Setting keywords to empty list.")
            parsed_keywords = []

        # Create a dictionary for consistent, processed output structure
        processed_result = {
            "sentiment": processed_sentiment,
            "explanation": processed_explanation,
            "keywords": parsed_keywords
        }

        logger.info(f"Processed DSPy result: {processed_result}")
        return processed_result # Return the processed dictionary

    except Exception as e:
        # Catch errors during the dspy.Predict call or subsequent processing
        logger.error(f"Error during DSPy prediction or processing: {e}", exc_info=True)
        # Return None to indicate failure
        return None


# --- Example Usage ---

review_positive = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the visuals were stunning. I left the theater feeling uplifted."
review_negative = "What a disappointment. The story was predictable, the characters were flat, and the pacing dragged terribly. I wouldn't recommend wasting your time on this."
review_neutral = "The film had some interesting concepts and decent cinematography, but the execution felt uneven. It wasn't bad, but it didn't leave a strong impression either."
review_tricky = "Visually, it's a masterpiece, truly groundbreaking effects. However, the narrative is a complete mess, making it hard to follow or care about."
review_empty = ""
review_non_movie = "This restaurant served the best pasta I've ever had."


def run_analysis(review_text: str, review_label: str):
    """Helper function to run and print analysis for a review."""
    print("\n" + "="*50)
    print(f"Analyzing {review_label} Review")
    print(f"Review Text: \"{review_text}\"")
    print("="*50)

    # --- Simple Prompting ---
    print("\n--- Simple Prompt Result ---")
    result_simple = analyze_sentiment_simple(review_text)
    # Use json.dumps for pretty printing the dictionary result
    print(json.dumps(result_simple, indent=2))

    # --- DSPy Prediction ---
    print("\n--- DSPy Predict Result ---")
    # Check if DSPy was set up correctly before calling
    if sentiment_predictor:
        result_dspy = analyze_sentiment_dspy(review_text)
        if result_dspy:
            # Print the processed dictionary result
            print(json.dumps(result_dspy, indent=2))
        else:
            print(" DSPy prediction failed or returned None.")
    else:
         print(" DSPy predictor was not initialized. Skipping DSPy analysis.")

# Run examples only if DSPy was configured (llm is not None)
# We run simple analysis regardless
run_analysis(review_positive, "Positive")
run_analysis(review_negative, "Negative")
run_analysis(review_neutral, "Neutral")
run_analysis(review_tricky, "Tricky")
run_analysis(review_empty, "Empty")
run_analysis(review_non_movie, "Non-Movie")


# --- Notes on DSPy Optimization ---
print("\n" + "="*50)
print("Notes on DSPy Optimization")
print("="*50)
print("""
The code above uses dspy.Predict, which is a basic module relying on the LLM's zero-shot capabilities
and the quality of the signature description. To potentially improve performance and robustness,
especially if the base model struggles, you can use DSPy's optimization features:

1.  **Create Training Data:** Collect a set of `dspy.Example` objects. Each example maps inputs
    (like `movie_review`) to the desired gold-standard outputs (`sentiment`, `explanation`, `keywords`).
    Mark the inputs using `.with_inputs('input_field_name')`.

    ```python
    # Example training data:
    # trainset = [
    #     dspy.Example(
    #         movie_review="Great film! Acting was top-notch.",
    #         sentiment="positive",
    #         explanation="The review praises the film and acting.",
    #         keywords=["great film", "acting", "top-notch"]
    #     ).with_inputs("movie_review"),
    #     dspy.Example(
    #         movie_review="Terrible plot, boring characters.",
    #         sentiment="negative",
    #         explanation="The review criticizes the plot and characters.",
    #         keywords=["terrible plot", "boring characters"]
    #     ).with_inputs("movie_review"),
    #     dspy.Example(
    #         movie_review="It was an okay movie, nothing special.",
    #         sentiment="neutral",
    #         explanation=None, # Or "" depending on preference
    #         keywords=["okay movie", "nothing special"]
    #     ).with_inputs("movie_review"),
    #     # ... add more diverse examples
    # ]
    ```

2.  **Define a Validation Metric:** Create a Python function that takes a gold `dspy.Example` and a predicted
    `dspy.Prediction` (or the output dictionary) and returns a score (typically float 0.0 to 1.0).
    The metric guides the optimizer by quantifying how well the predictions match the desired outputs.

    ```python
    # Example metric (can be more sophisticated):
    # def validate_sentiment_analysis(gold, pred, trace=None):
    #     # Ensure pred is the dictionary output from analyze_sentiment_dspy
    #     # or access fields like pred.sentiment if using the raw Prediction object
    #     pred_sentiment = pred.get('sentiment') if isinstance(pred, dict) else getattr(pred, 'sentiment', None)
    #     pred_explanation = pred.get('explanation') if isinstance(pred, dict) else getattr(pred, 'explanation', None)
    #     pred_keywords = pred.get('keywords', []) if isinstance(pred, dict) else getattr(pred, 'keywords', [])
    #
    #     # Gold standard values
    #     gold_sentiment = gold.sentiment
    #     gold_explanation = gold.explanation # Assume None/empty string consistency
    #     gold_keywords = gold.keywords
    #
    #     # Score calculation (example: 1 point for each correct field)
    #     score = 0.0
    #     max_score = 3.0
    #
    #     # Sentiment match
    #     if pred_sentiment == gold_sentiment:
    #         score += 1.0
    #
    #     # Explanation match (handle None/empty string for neutral)
    #     if gold_sentiment == "neutral":
    #         if pred_explanation is None or pred_explanation == "":
    #             score += 1.0
    #     elif pred_explanation == gold_explanation: # Simplistic check for non-neutral
    #         score += 1.0
    #
    #     # Keyword match (example: Jaccard index or simple set comparison)
    #     if isinstance(pred_keywords, list) and isinstance(gold_keywords, list):
    #          pred_kw_set = set(pred_keywords)
    #          gold_kw_set = set(gold_keywords)
    #          if pred_kw_set == gold_kw_set: # Exact match for simplicity
    #               score += 1.0
    #          # Could use intersection/union for partial credit
    #
    #     return score / max_score # Normalize score
    ```

3.  **Choose an Optimizer (Teleprompter):** `dspy.BootstrapFewShot` is common for optimizing prompts by selecting
    good examples from the training set to include in the prompt (few-shot learning).

    ```python
    # from dspy.teleprompt import BootstrapFewShot
    #
    # # Configure the teleprompter with the metric
    # # max_bootstrapped_demos controls how many examples are included in the optimized prompt
    # teleprompter = BootstrapFewShot(metric=validate_sentiment_analysis, max_bootstrapped_demos=3)
    ```

4.  **Define and Compile a DSPy Module:** Wrap your predictor(s) in a `dspy.Module`. The `compile` method
    runs the optimization process using the teleprompter, metric, and training data.

    ```python
    # class SentimentAnalysisModule(dspy.Module):
    #      def __init__(self):
    #          super().__init__()
    #          # You could potentially use more complex DSPy modules here,
    #          # like dspy.ChainOfThought(SentimentAnalysisSignature)
    #          self.predictor = dspy.Predict(SentimentAnalysisSignature)
    #
    #      def forward(self, movie_review):
    #          # The forward method defines how data flows through the module
    #          prediction_obj = self.predictor(movie_review=movie_review)
    #          # You might perform post-processing here if needed, similar to analyze_sentiment_dspy
    #          # For simplicity, just return the prediction object
    #          return prediction_obj
    #
    # # Instantiate the module
    # module_to_optimize = SentimentAnalysisModule()
    #
    # # Compile the module (this runs the LLM multiple times on the trainset)
    # # Ensure trainset and validate_sentiment_analysis are defined
    # # optimized_sentiment_analyzer = teleprompter.compile(module_to_optimize, trainset=trainset)
    ```

5.  **Use the Optimized Module:** Make predictions using the compiled module. It should now incorporate
    optimized prompts (potentially including few-shot examples).

    ```python
    # # Make predictions with the optimized module
    # # result_optimized = optimized_sentiment_analyzer(movie_review="Another review to analyze...")
    # # Process the output 'result_optimized' as needed (it will be a dspy.Prediction object)
    ```

Optimization requires a good dataset and a meaningful metric but allows DSPy to automatically tailor the interaction with the LLM for better performance and reliability on your specific task.
""")

