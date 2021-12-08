import boto3
import json
from utilities import vectorize_sequences, one_hot_encode
client = boto3.client('sagemaker-runtime')

vocabulary_length = 9013

test_messages = ["hey there buddy"]
one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

response = client.invoke_endpoint(
    EndpointName='sms-spam-classifier-mxnet-2021-12-08-05-09-31-356',
    Body=encoded_test_messages,
    ContentType='application/json'
)

result = response['Body'].read()
data = json.loads(result)
print(data)
"""
from sagemaker.mxnet.model import MXNetPredictor
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
vocabulary_length = 9013

# Uncomment the following line to connect to an existing endpoint.
mxnet_pred = MXNetPredictor('sms-spam-classifier-mxnet-2021-12-08-05-09-31-356')

test_messages = ["Hey whats up?"]
one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

result = mxnet_pred.predict(encoded_test_messages)
print(result)
"""