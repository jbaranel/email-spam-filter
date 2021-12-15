from utilities import one_hot_encode, vectorize_sequences
import boto3
import email
import re
import json
import os


def make_prediction(message):
    vocabulary_length = 9013
    runtime = boto3.client('runtime.sagemaker')
    ENDPOINT = os.environ['SAGEMAKER_ENDPOINT']
    
    test_messages = [message]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='application/json',
        Body=json.dumps(encoded_test_messages.tolist()))

    result = json.loads(response['Body'].read().decode())

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

    date_received = b["date"]
    subject = b["subject"]    
    sender = re.findall(r"\<(.*?)\>", b["from"])[0]
    
    body = b.get_payload(decode=True)   
    body = body.decode()
    email_body = body.replace('\n', ' ').replace('\r', '').strip(' ')[0:240]

    results = {
        'date_received': date_received,
        'subject': subject,
        'body': email_body,
        'sender': sender
    }
    return results


def return_results(classification, confidence_score, email):
    ses_client = boto3.client('ses')
    
    r1 = f"We received your email sent at {email['date_received']} with the subject {email['subject']}."
    r2 = f"Here is a 240 character sample of the email body: {email['body']}\r\n"
    r3 = f"The email was categorized as {classification} with a {confidence_score}% confidence."

    SUBJECT = "Spam filter results"

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = f"""        
        {r1}\n
        {r2}\n
        {r3}\n
        """
    # The HTML body of the email.
    BODY_HTML = f"""
        <html>
        <head></head>
        <body>
            <h1>Spam Results</h1>
            <p>{r1}</p>
            <p>{r2}</p>
            <p>{r3}</p>
        </body>
        </html>
        """

    CHARSET = "UTF-8"

    receiver_email = email['sender']
    response = ses_client.send_email(
        Source = 'no-reply@spam-filter-jb7607.com',
        Destination = {
            'ToAddresses': [
                receiver_email
            ]
        },
        Message = {
            'Body': {
                'Html': {
                    'Charset': CHARSET,
                    'Data': BODY_HTML,
                },
                'Text': {
                    'Charset': CHARSET,
                    'Data': BODY_TEXT,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': SUBJECT,
            },
        }
    )
    print(response)
    

def lambda_handler(event, context):
    s3_event = event['Records'][0]['s3']
    bucket = s3_event['bucket']['name']
    key = s3_event['object']['key']    
    email_results = email_parse(bucket, key)
    classification, confidence_score = make_prediction(email_results['body'])
    return_results(classification, confidence_score, email_results)

