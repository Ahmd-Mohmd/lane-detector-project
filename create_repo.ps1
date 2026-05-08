$token = "YOUR_GITHUB_TOKEN_HERE"
$username = "Ahmd-Mohmd"
$email = "ahmedabdelsalam1234567@gmail.com"

$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
    "User-Agent" = "PowerShell"
}

$body = @{
    name = "lane-detector"
    description = "Lane detection project"
    private = $false
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json" -SkipHttpErrorCheck

Write-Host "Status Code: $($response.StatusCode)"
Write-Host "Response Content:"
Write-Host $response.Content
