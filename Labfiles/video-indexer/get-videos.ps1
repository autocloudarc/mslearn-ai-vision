$account_id="7f6168d7-5528-454b-8ec4-d8bc5ea4dda2"
$api_key="4241c91e2f654466a41b30b054289545"
$location="trial"

# Clear the console window
cls

# Call the AccessToken method with the API key in the header to get an access token
$token = Invoke-RestMethod -Method "Get" -Uri "https://api.videoindexer.ai/auth/$location/Accounts/$account_id/AccessToken" -Headers @{'Ocp-Apim-Subscription-Key' = $api_key}

# Use the access token to make an authenticated call to the Videos method to get a list of videos in the account
Invoke-RestMethod -Method "Get" -Uri "https://api.videoindexer.ai/$location/Accounts/$account_id/Videos?accessToken=$token" | ConvertTo-Json -Depth 6
