require 'net/http'
require 'json'

req = {
    prompt: 'The first man to',
    max_tokens: 20
}

url = URI.parse 'http://localhost:7331/complete'

result = Net::HTTP.post url, JSON.generate(req), 'Content-Type' => 'text/json'

p result.each_header.to_h
puts result.body
