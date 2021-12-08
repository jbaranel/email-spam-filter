from sagemaker.mxnet.model import MXNetPredictor
from utilities import one_hot_encode, vectorize_sequences
import boto3
import email
from email import policy

def make_prediction(message):
    vocabulary_length = 9013
    mxnet_pred = MXNetPredictor('sms-spam-classifier-mxnet-2021-12-08-05-09-31-356')

    test_messages = [message]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    result = mxnet_pred.predict(encoded_test_messages)
    confidence_score = round(result['predicted_probability'][0][0] * 100, 2)
    label = result['predicted_label'][0][0]
    if label == 1.0:
        classification = "Spam"
    else:
        classification = "Not Spam"

    return [classification, confidence_score]

def email_parse(bucket, key):
    s3_client = boto3.resource('s3')    
    email_obj = s3_client.Object(bucket, key)
    email_body = email_obj.get()['Body'].read()
    b = email.message_from_bytes(email_body)
    body = ""

    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = b.get_payload(decode=True)

    body = body.decode()
    email_body = body.replace('\n', ' ').replace('\r', '').strip(' ')
    return email_body


def return_results(classification, confidence_score):
    email_response_body = f"""
    We received your email sent at [EMAIL_RECEIVE_DATE] with the subject [EMAIL_SUBJECT].\n
    Here is a 240 character sample of the email body: [EMAIL_BODY]\n
    The email was categorized as {classification} with a {confidence_score}% confidence."""
    print(email_response_body)
    

def lambda_handler(event, context):
    bucket = 'email-received-bucket'
    key = '30gma38vv32jg36culehfo9orct7mrqnv2uiu481'
    email_body = email_parse(bucket, key)
    classification, confidence_score = make_prediction(email_body)
    return_results(classification, confidence_score)

lambda_handler({}, {})
