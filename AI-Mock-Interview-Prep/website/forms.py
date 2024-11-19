from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField

class InterviewForm(FlaskForm):
    answer = TextAreaField('Your Answer:')
    submit_answer = SubmitField('Submit Answer')
    generate_questions = SubmitField('Generate A Question')
