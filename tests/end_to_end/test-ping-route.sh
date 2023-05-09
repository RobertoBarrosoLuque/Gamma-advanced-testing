#!/bin/bash

PING_RESPONSE=$(curl 'http://10.16.40.21:5000/ping')

echo $PING_RESPONSE

if [[ $PING_RESPONSE == pong! ]];
then
  echo "Test PASSED!"
else
  exit "Test FAILED!"
fi