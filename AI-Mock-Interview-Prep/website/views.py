from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import subprocess
import sys
from flask_login import login_required, current_user
from . forms import InterviewForm
import random
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


views = Blueprint('views', __name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

feedback_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
feedback_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

qa_pipeline = pipeline("text-generation", model="gpt2")  # GPT-2 for text generation
sentiment_analysis_pipeline = pipeline("sentiment-analysis")  # Sentiment analysis

def generated_behavioral_questions():
    prompt = "Behavioral interview questions: "
    result = qa_pipeline(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']


def analyze_response(answer):
    sentiment = sentiment_analysis_pipeline(answer)
    return sentiment

@views.route('/')
def home():
    return render_template("home.html", user=current_user)

@views.route('/about/')
def about():
    return render_template("about.html", user=current_user)
 
@views.route('/profile/')
@login_required
def profile():
    return render_template("profile.html", user=current_user)

@views.route('/behavioral/')
def bev():
    return render_template("behavorial.html", user=current_user)

@views.route('/technical/')
def tech():
    return render_template("technical.html", user=current_user)

@views.route('/editor/')
def edit():
    return render_template("editor.html", user=current_user)

@views.route('/run', methods=['POST'])
def run_code():
    data = request.get_json()
    code = data.get('code')
    
    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],  # Use sys.executable to dynamically get Python path
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return jsonify({"output": output})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@views.route('/feedback', methods=["GET", "POST"])
def feedback():
    form = InterviewForm()
    generated_question = None
    analysis_feedback = None
    real_feedback = None

    if request.method == "POST":
        # Handle Question Generation
        if 'generate_questions' in request.form:
            questions = [
                "Tell me about a time you faced a conflict at work.",
                "Describe a time you showed leadership.",
                "How do you prioritize tasks under tight deadlines?",
                "What’s an example of a mistake you made and how you handled it?",
                "Tell me about a time you worked on a team project.",
                "How do you handle constructive criticism?",
                "Describe a time you disagreed with your manager.",
                "How do you approach learning a new skill?",
                "Tell me about a time you went above and beyond at work.",
                "Describe how you’ve dealt with a difficult coworker."
            ]
            generated_question = random.choice(questions)

        # Handle Answer Submission
        elif 'submit_answer' in request.form:
            user_answer = form.answer.data

            # Validate Input
            if not user_answer.strip():
                flash("Please provide an answer before submitting.", "danger")
                return redirect(url_for('feedback'))

            try:
                # Sentiment Analysis
                inputs = tokenizer(user_answer, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

                labels = ["Negative", "Positive"]
                analysis_feedback = [
                    {"label": labels[i], "score": predictions[0][i].item()}
                    for i in range(len(labels))
                ]

                # Generate AI Feedback - Pass the question along with the answer to improve relevance
                feedback_prompt = f"Question: {generated_question} User's answer: '{user_answer}'. How can the answer be improved based on the question?"

                # Tokenize the feedback prompt and handle token limit
                input_ids = feedback_tokenizer.encode(feedback_prompt, return_tensors="pt", truncation=True, max_length=512)

                # Generate the feedback
                feedback_output = feedback_model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)
                real_feedback = feedback_tokenizer.decode(feedback_output[0], skip_special_tokens=True)

                # Ensure feedback is not just the input prompt
                if real_feedback.strip() == feedback_prompt.strip():
                    real_feedback = "Sorry, the feedback generation didn't work. Please try again."

            except Exception as e:
                # Log the error and show a user-friendly message
                print(f"Error during processing: {e}")
                flash("Something went wrong while analyzing your response. Please try again.", "danger")
                return redirect(url_for('feedback'))

    return render_template(
        "feedback.html", user=current_user,
        form=form,
        generated_question=generated_question,
        analysis_feedback=analysis_feedback,
        real_feedback=real_feedback
    )