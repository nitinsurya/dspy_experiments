# Ensure you have the necessary libraries installed:
# pip install dspy-ai litellm openai PyYAML # openai is needed by dspy, PyYAML can be helpful
# Example: pip install dspy-ai==2.4.3 litellm==1.37.10 openai==1.28.1 PyYAML==6.0.1
# NOTE: dspy.Ollama requires a recent version of dspy-ai. If you get an AttributeError,
# try upgrading: pip install --upgrade dspy-ai

import dspy
import litellm
import os
import json
import logging
import ast # For safely evaluating string representations of lists
from typing import List, Optional, Dict, Any
from dspy.teleprompt import BootstrapFewShot # Import the optimizer

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

# --- Approach 1: Simple Prompting with litellm (Function remains the same) ---

def analyze_sentiment_simple(review: str) -> Dict[str, Any]:
    """
    Analyzes sentiment using a direct prompt with litellm.
    (Implementation is the same as the previous version - kept for comparison)
    """
    logger.info("\n--- Running Simple Prompting ---")
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
""" # End of the f-string

    try:
        logger.info("Sending request to Ollama via litellm...")
        # litellm.set_verbose = True # Uncomment for debugging
        response = litellm.completion(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            api_base=OLLAMA_API_BASE,
            temperature=0.1,
            max_tokens=200,
            request_timeout=60
        )
        # litellm.set_verbose = False

        logger.info("Received response from Ollama.")
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             logger.error("Error: Received empty or invalid response structure from litellm.")
             return {"error": "Received empty response from LLM"}

        content = response.choices[0].message.content.strip()
        logger.info(f"Raw response content:\n{content}")

        result = None
        try:
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = content[json_start:json_end+1]
                logger.info(f"Extracted JSON string: {json_string}")
                result = json.loads(json_string)
            else:
                 logger.error("Could not find valid JSON object delimiters {{...}} in the response.")
                 raise json.JSONDecodeError("No JSON object found", content, 0)

            required_keys = ["sentiment", "explanation", "keywords"]
            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing required keys in JSON response: {missing_keys}")

            valid_sentiments = ["positive", "negative", "neutral"]
            if result.get("sentiment") not in valid_sentiments:
                raise ValueError(f"Invalid sentiment value: '{result.get('sentiment')}'")

            if not isinstance(result.get("keywords"), list):
                 raise ValueError(f"Keywords should be a list, got: {type(result.get('keywords'))}")

            if result["sentiment"] == "neutral" and result.get("explanation") is not None:
                 logger.warning("Explanation provided for neutral sentiment, setting to None.")
                 result["explanation"] = None
            elif result["sentiment"] != "neutral" and result.get("explanation") is None:
                 logger.warning("Explanation missing (is null) for non-neutral sentiment. Setting to empty string.")
                 result["explanation"] = ""

            logger.info("Successfully parsed and validated JSON response.")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Error: Failed to parse JSON response: {e}")
            logger.error(f"Content attempted to parse: {json_string if 'json_string' in locals() else content}")
            return {"error": "Failed to parse LLM response as JSON", "raw_content": content}
        except ValueError as e:
            logger.error(f"Error: Invalid JSON structure or values: {e}")
            logger.error(f"Parsed content (if available) before error: {result}")
            return {"error": f"Invalid JSON structure or value: {e}", "parsed_content_before_error": result}

    except litellm.exceptions.APIConnectionError as e:
         logger.error(f"Error: Connection to Ollama API failed at {OLLAMA_API_BASE}: {e}", exc_info=True)
         return {"error": f"API Connection Error: {e}"}
    except litellm.exceptions.Timeout as e:
         logger.error(f"Error: Request to Ollama timed out: {e}", exc_info=True)
         return {"error": f"API Timeout Error: {e}"}
    except Exception as e:
        logger.error(f"Error during litellm completion call: {e}", exc_info=True)
        return {"error": f"LLM API call failed unexpectedly: {e}"}


# --- Approach 2: DSPy with Optimization ---

# 1. Define the Signature (Same as before)
class SentimentAnalysisSignature(dspy.Signature):
    """Analyzes movie review sentiment, providing explanation and keywords.
    Input: A movie review.
    Output: Sentiment (positive/negative/neutral), explanation (if applicable), and keywords (as a list)."""

    movie_review = dspy.InputField(desc="The text of the movie review.")
    sentiment = dspy.OutputField(desc="The overall sentiment (must be one of 'positive', 'negative', or 'neutral').")
    explanation = dspy.OutputField(desc="A brief explanation for positive/negative sentiment (should be null or empty string if sentiment is neutral).")
    keywords = dspy.OutputField(desc="A Python-style list of strings (e.g., ['keyword1', 'keyword2']) representing keywords related to the sentiment and movie aspects.")

# 2. Define Training Examples (`EXAMPLES`)
# Create a list of dspy.Example objects for training/optimization
# Mark the inputs using .with_inputs()
EXAMPLES = [
    dspy.Example(
        movie_review="Absolutely loved it! The acting was incredible and the story kept me hooked.",
        sentiment="positive",
        explanation="The reviewer expresses strong positive sentiment, highlighting acting and story.",
        keywords=["loved it", "incredible acting", "hooked", "story"]
    ).with_inputs("movie_review"),
    dspy.Example(
        movie_review="A complete waste of time. The plot was nonsensical and the characters were wooden.",
        sentiment="negative",
        explanation="The reviewer expresses strong negative sentiment, criticizing the plot and characters.",
        keywords=["waste of time", "nonsensical plot", "wooden characters"]
    ).with_inputs("movie_review"),
    dspy.Example(
        movie_review="It was okay. Some good visuals, but the pacing felt off and it didn't leave much impact.",
        sentiment="neutral",
        explanation=None, # Explanation should be None or empty for neutral
        keywords=["okay", "good visuals", "pacing off", "no impact"]
    ).with_inputs("movie_review"),
    dspy.Example(
        movie_review="Visually stunning, a technical marvel! But the script was weak and confusing.",
        sentiment="negative", # Leaning negative due to script criticism, could also be argued as neutral/mixed
        explanation="Despite praising visuals, the reviewer criticizes the weak and confusing script.",
        keywords=["visually stunning", "technical marvel", "weak script", "confusing"]
    ).with_inputs("movie_review"),
     dspy.Example(
        movie_review="A heartwarming story with brilliant performances. A must-see!",
        sentiment="positive",
        explanation="The review highlights the heartwarming story and brilliant performances, recommending the movie.",
        keywords=["heartwarming story", "brilliant performances", "must-see"]
    ).with_inputs("movie_review"),
    dspy.Example(
        movie_review="I fell asleep halfway through. Incredibly boring.",
        sentiment="negative",
        explanation="The reviewer found the movie extremely boring, indicating strong negative sentiment.",
        keywords=["fell asleep", "boring"]
    ).with_inputs("movie_review"),
     dspy.Example(
        movie_review="The cinematography was decent and the soundtrack was nice, but otherwise unremarkable.",
        sentiment="neutral",
        explanation=None,
        keywords=["decent cinematography", "nice soundtrack", "unremarkable"]
    ).with_inputs("movie_review"),
]
logger.info(f"Defined {len(EXAMPLES)} training examples.")

# 3. Define Validation Metric
def validate_sentiment_analysis(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Validates the predicted sentiment analysis against the gold standard.

    Args:
        gold: The gold standard dspy.Example.
        pred: The predicted dspy.Prediction object.
        trace: Optional trace object (unused here).

    Returns:
        A score between 0.0 and 1.0 indicating the quality of the prediction.
    """
    # Extract predicted values safely using getattr
    pred_sentiment = getattr(pred, 'sentiment', None)
    pred_explanation = getattr(pred, 'explanation', None)
    raw_pred_keywords = getattr(pred, 'keywords', None)

    # Extract gold values
    gold_sentiment = gold.sentiment
    gold_explanation = gold.explanation # Assumes None/empty string consistency in EXAMPLES
    gold_keywords = gold.keywords

    # --- Score Calculation ---
    score = 0.0
    max_score = 3.0 # One point for each field: sentiment, explanation, keywords

    # 1. Sentiment match
    if pred_sentiment == gold_sentiment:
        score += 1.0
        logger.debug(f"Metric: Sentiment matched ({pred_sentiment})")
    else:
        logger.debug(f"Metric: Sentiment MISMATCH (Pred: {pred_sentiment}, Gold: {gold_sentiment})")


    # 2. Explanation match (handle None/empty string for neutral)
    # Normalize explanations (treat None, '', 'null', 'None' as equivalent for comparison)
    def normalize_explanation(expl):
        if expl is None or str(expl).strip().lower() in ['null', 'none', '']:
            return None
        return str(expl).strip()

    norm_pred_explanation = normalize_explanation(pred_explanation)
    norm_gold_explanation = normalize_explanation(gold_explanation)

    if norm_pred_explanation == norm_gold_explanation:
        score += 1.0
        logger.debug(f"Metric: Explanation matched ({norm_pred_explanation})")
    else:
        logger.debug(f"Metric: Explanation MISMATCH (Pred: {norm_pred_explanation}, Gold: {norm_gold_explanation})")


    # 3. Keyword match (using set comparison for flexibility in order)
    pred_keywords_list = []
    if isinstance(raw_pred_keywords, list):
        pred_keywords_list = [str(k).strip() for k in raw_pred_keywords]
    elif isinstance(raw_pred_keywords, str):
        # Attempt to parse if it's a string representation of a list
        try:
            evaluated = ast.literal_eval(raw_pred_keywords)
            if isinstance(evaluated, list):
                pred_keywords_list = [str(k).strip() for k in evaluated]
            else: # Treat as single keyword if parsing fails or result isn't list
                 pred_keywords_list = [raw_pred_keywords.strip()] if raw_pred_keywords.strip() else []
        except: # Handle parsing errors
             pred_keywords_list = [k.strip() for k in raw_pred_keywords.split(',') if k.strip()] # Fallback to comma split
             if not pred_keywords_list and raw_pred_keywords.strip():
                  pred_keywords_list = [raw_pred_keywords.strip()]


    # Ensure gold keywords are strings and create sets
    gold_keywords_set = set(str(k).strip() for k in gold_keywords)
    pred_keywords_set = set(pred_keywords_list)

    # Simple exact set match for score (could use Jaccard index for partial credit)
    if gold_keywords_set == pred_keywords_set:
        score += 1.0
        logger.debug(f"Metric: Keywords matched ({pred_keywords_set})")
    else:
        logger.debug(f"Metric: Keywords MISMATCH (Pred: {pred_keywords_set}, Gold: {gold_keywords_set})")


    final_score = score / max_score
    logger.debug(f"Metric: Final score for example = {final_score}")
    return final_score


# 4. Configure the Language Model (Same as before, ensure dspy version is compatible)
llm = None
ollama_model_id = OLLAMA_MODEL_NAME.split('/')[-1] if '/' in OLLAMA_MODEL_NAME else OLLAMA_MODEL_NAME
# --- IMPORTANT: Requires recent dspy-ai version ---
llm = dspy.LM(
    model=DSPY_OLLAMA_MODEL_NAME,
    api_base=OLLAMA_API_BASE,
    max_tokens=200, # Consistent with simple approach
    temperature=0.1,
    request_timeout=60 # Add timeout
)
dspy.settings.configure(lm=llm)
logger.info(f"DSPy configured with Ollama model '{ollama_model_id}' at {OLLAMA_API_BASE}.")

# 5. Define the DSPy Module
class SentimentAnalysisModule(dspy.Module):
     def __init__(self):
         super().__init__()
         # Using Predict, but could use ChainOfThought for more complex reasoning
         self.predictor = dspy.Predict(SentimentAnalysisSignature)

     def forward(self, movie_review):
         # The forward method defines how data flows through the module
         prediction = self.predictor(movie_review=movie_review)
         # We return the raw prediction object here; metric function handles extraction
         return prediction

# 6. Configure the Optimizer (Teleprompter)
if llm: # Only configure optimizer if LLM is set up
    teleprompter = BootstrapFewShot(
        metric=validate_sentiment_analysis,
        max_bootstrapped_demos=3, # Number of examples optimizer will try to include in the prompt
        max_labeled_demos=10, # Max examples provided to the optimizer's LLM call
        max_rounds=3 # Number of optimization rounds
    )
    logger.info("Configured BootstrapFewShot optimizer.")
else:
    teleprompter = None
    raise Exception("Skipping optimizer configuration due to LLM setup failure.")

# 7. Compile the Module
compiled_sentiment_analyzer = None
if llm and teleprompter:
    try:
        logger.info("Starting DSPy module compilation...")
        # Instantiate the module
        module_to_optimize = SentimentAnalysisModule()
        # Compile the module using the optimizer and training examples
        # This step involves multiple LLM calls to find good prompt examples
        compiled_sentiment_analyzer = teleprompter.compile(module_to_optimize, trainset=EXAMPLES)
        logger.info("DSPy module compilation finished.")

        # --- Inspecting Prompts and History ---
        logger.info("\n--- Inspecting LLM History from Compilation ---")
        # Access the history of LLM calls made during compilation
        # Note: History tracking might need to be explicitly enabled in some LM configs.
        # dspy.Ollama (via litellm) usually captures it.
        if hasattr(llm, 'history') and llm.history:
             print(f"Number of LLM calls during compilation: {len(llm.history)}")
             # Print details of the last few calls (can be very verbose)
             num_calls_to_show = 5
             print(f"Showing details for the last {num_calls_to_show} calls:")
             for i, call in enumerate(llm.history[-num_calls_to_show:]):
                  print(f"\n--- Call {len(llm.history) - num_calls_to_show + i + 1} ---")
                  # **MODIFIED: Print messages instead of prompt**
                  messages = call.get('messages', 'N/A')
                  print("Messages Sent:")
                  if isinstance(messages, list):
                      for msg in messages:
                          print(f"  Role: {msg.get('role')}, Content: {msg.get('content')[:200]}...") # Truncate long content
                  else:
                      print(f"  {messages}") # Print as is if not a list

                  print(f"\nResponse:\n{call.get('response', 'N/A')}")
                  # print(f"\nkwargs: {call.get('kwargs', 'N/A')}") # Often less critical
                  print(f"Timestamp: {call.get('timestamp', 'N/A')}")
                  # Metrics aren't stored *in* the history call object itself.
                  # They are computed *by* the teleprompter using the results of these calls.
        else:
             print("LLM history not available or empty. Ensure history tracking is enabled if needed.")

        # You can also inspect the compiled module's predictor to see the final prompt structure
        if compiled_sentiment_analyzer:
             try:
                  # Accessing the predictor within the compiled module
                  final_predictor = compiled_sentiment_analyzer.predictor
                  print("\n--- Final Compiled Predictor Signature Instructions ---")
                  # This shows the base instructions part of the prompt
                  print(final_predictor.signature.instructions)
                  print("\n--- Final Compiled Predictor Few-Shot Examples ---")
                  # These are the examples selected by the optimizer to guide the LLM
                  if hasattr(final_predictor, 'demos') and final_predictor.demos:
                      print(f"Number of selected demos: {len(final_predictor.demos)}")
                      for demo_num, demo in enumerate(final_predictor.demos):
                          print(f"\n--- Demo {demo_num + 1} ---")
                          print(f"Inputs: {demo.inputs()}")
                          print(f"Outputs: {demo.outputs()}")
                  else:
                      print("No few-shot examples were selected or stored by the optimizer in this predictor.")

             except Exception as e:
                  logger.warning(f"Could not inspect final compiled predictor details: {e}")

    except Exception as e:
        logger.error(f"Error during DSPy module compilation: {e}", exc_info=True)
        compiled_sentiment_analyzer = None # Ensure it's None if compilation fails
else:
    logger.error("Skipping compilation because LLM or optimizer setup failed.")

# exit()

# --- Function to run analysis using OPTIMIZED module ---
# (Similar to analyze_sentiment_dspy but uses the compiled module)
def analyze_sentiment_dspy_optimized(review: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes sentiment using the *compiled* DSPy module.

    Args:
        review: The movie review text.

    Returns:
        A dictionary containing processed sentiment, explanation, and keywords,
        or None if the compiled module is not available or prediction fails.
    """
    if not compiled_sentiment_analyzer:
        logger.error("Compiled DSPy module is not available. Cannot analyze sentiment.")
        return None

    logger.info("\n--- Running DSPy Optimized Predict ---")
    logger.info("Sending request via Compiled DSPy Module...")

    try:
        # Use the compiled module for prediction
        prediction = compiled_sentiment_analyzer(movie_review=review)
        logger.info("Received prediction object from Compiled DSPy Module.")
        logger.debug(f"Raw DSPy prediction sentiment: {getattr(prediction, 'sentiment', 'MISSING')}")
        logger.debug(f"Raw DSPy prediction explanation: {getattr(prediction, 'explanation', 'MISSING')}")
        logger.debug(f"Raw DSPy prediction keywords: {getattr(prediction, 'keywords', 'MISSING')}")

        # --- Validation and Cleanup (Identical to analyze_sentiment_dspy) ---
        processed_sentiment = getattr(prediction, 'sentiment', None)
        processed_explanation = getattr(prediction, 'explanation', None)
        raw_keywords = getattr(prediction, 'keywords', None)
        parsed_keywords = []

        valid_sentiments = ["positive", "negative", "neutral"]
        if processed_sentiment not in valid_sentiments:
             logger.warning(f"Optimized: Received unexpected or missing sentiment: '{processed_sentiment}'. Setting to None.")
             processed_sentiment = None

        def normalize_explanation(expl):
            if expl is None or str(expl).strip().lower() in ['null', 'none', '']: return None
            return str(expl).strip()

        norm_pred_explanation = normalize_explanation(processed_explanation)
        if processed_sentiment == "neutral":
            if norm_pred_explanation is not None:
                logger.warning(f"Optimized: Explanation '{processed_explanation}' provided for neutral sentiment. Clearing.")
                norm_pred_explanation = None
        elif processed_sentiment in ["positive", "negative"]:
            if norm_pred_explanation is None:
                logger.warning(f"Optimized: Explanation missing for non-neutral sentiment '{processed_sentiment}'. Setting to empty.")
                norm_pred_explanation = ""

        if isinstance(raw_keywords, list):
            parsed_keywords = [str(k).strip() for k in raw_keywords if isinstance(k, str)] # Ensure strings
        elif isinstance(raw_keywords, str):
            kw_str = raw_keywords.strip()
            if kw_str:
                try:
                    evaluated = ast.literal_eval(kw_str)
                    if isinstance(evaluated, list):
                         parsed_keywords = [str(k).strip() for k in evaluated if isinstance(k, str)]
                    else:
                         parsed_keywords = [kw_str]
                except:
                     parsed_keywords = [k.strip() for k in kw_str.split(',') if k.strip()]
                     if not parsed_keywords: parsed_keywords = [kw_str]
        elif raw_keywords is not None:
             parsed_keywords = [str(raw_keywords)]


        processed_result = {
            "sentiment": processed_sentiment,
            "explanation": norm_pred_explanation, # Use normalized explanation
            "keywords": parsed_keywords
        }

        logger.info(f"Processed Optimized DSPy result: {processed_result}")
        return processed_result

    except Exception as e:
        logger.error(f"Error during Optimized DSPy prediction or processing: {e}", exc_info=True)
        return None


# --- Example Usage ---

# Define example reviews (same as before)
review_positive = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the visuals were stunning. I left the theater feeling uplifted."
review_negative = "What a disappointment. The story was predictable, the characters were flat, and the pacing dragged terribly. I wouldn't recommend wasting your time on this."
review_neutral = "The film had some interesting concepts and decent cinematography, but the execution felt uneven. It wasn't bad, but it didn't leave a strong impression either."
review_tricky = "Visually, it's a masterpiece, truly groundbreaking effects. However, the narrative is a complete mess, making it hard to follow or care about."
review_empty = ""
review_non_movie = "This restaurant served the best pasta I've ever had."

# Updated run_analysis function to include optimized results
def run_analysis(review_text: str, review_label: str):
    """Helper function to run and print analysis for a review."""
    print("\n" + "="*80)
    print(f"Analyzing {review_label} Review")
    print(f"Review Text: \"{review_text}\"")
    print("="*80)

    # --- Simple Prompting ---
    print("\n--- Simple Prompt Result ---")
    result_simple = analyze_sentiment_simple(review_text)
    print(json.dumps(result_simple, indent=2))

    # --- DSPy Optimized Prediction ---
    print("\n--- DSPy Optimized Predict Result ---")
    if compiled_sentiment_analyzer: # Check if compiled module exists
        result_dspy_opt = analyze_sentiment_dspy_optimized(review_text)
        if result_dspy_opt:
            print(json.dumps(result_dspy_opt, indent=2))
        else:
            print(" DSPy Optimized prediction failed or returned None.")
    else:
         print(" Compiled DSPy module is not available. Skipping Optimized DSPy analysis.")

# Run examples
run_analysis(review_positive, "Positive")
run_analysis(review_negative, "Negative")
run_analysis(review_neutral, "Neutral")
run_analysis(review_tricky, "Tricky")
run_analysis(review_empty, "Empty")
run_analysis(review_non_movie, "Non-Movie")


# --- Notes on DSPy Optimization ---
# (Notes section remains largely the same, emphasizing the process)
print("\n" + "="*80)
print("Notes on DSPy Optimization Process Used Above")
print("="*80)
print("""
The script now demonstrates the DSPy optimization workflow:

1.  **Training Data (`EXAMPLES`):** A list of `dspy.Example` objects was created, providing input reviews and their desired structured outputs (sentiment, explanation, keywords).
2.  **Validation Metric (`validate_sentiment_analysis`):** A function was defined to compare a prediction against a gold example, returning a score from 0.0 to 1.0. This function handles nuances like comparing explanations (treating None/empty as equivalent for neutral) and keyword sets.
3.  **Optimizer (`BootstrapFewShot`):** This teleprompter was configured with the validation metric. During compilation, it uses the metric to evaluate different combinations of examples from `EXAMPLES` to find few-shot examples that improve performance on the training set.
4.  **DSPy Module (`SentimentAnalysisModule`):** The `dspy.Predict` logic was wrapped in a `dspy.Module`. This is necessary for the `compile` process.
5.  **Compilation (`teleprompter.compile(...)`):** This crucial step executed the optimization. The optimizer ran the `SentimentAnalysisModule` on the training examples, evaluated the outputs using the metric, and selected the best few-shot examples to include in the final prompt template used by the `compiled_sentiment_analyzer`. The LLM history during this phase shows the internal trials.
6.  **Inference (`analyze_sentiment_dspy_optimized`):** Predictions were made using the `compiled_sentiment_analyzer`. This module now potentially uses an optimized prompt structure (including few-shot examples selected during compilation) which can lead to more accurate and consistent results compared to the zero-shot `dspy.Predict` or the simple prompting approach.

This process allows DSPy to adapt the prompting strategy to the specific task and LLM, potentially overcoming weaknesses of zero-shot prompting.
""")
