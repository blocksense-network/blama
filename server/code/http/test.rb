require 'net/http'
require 'json'

SERVER = 'http://localhost:7331'

completeReq = {
    prompt: 'The first man to',
    max_tokens: 20
}

puts "running..."
result = Net::HTTP.post URI.parse(SERVER + '/complete'), JSON.generate(completeReq), 'Content-Type' => 'text/json'

p result.each_header.to_h
puts result.body

verifyReq = {
    request: completeReq,
    response: JSON::parse(result.body)
}

puts "verifying..."
result = Net::HTTP.post URI.parse(SERVER + '/verify_completion'), JSON.generate(verifyReq), 'Content-Type' => 'text/json'
p result.each_header.to_h
puts result.body

# chatCompleteReq = {
#     messages: [
#         { role: 'system', content: 'You are a helpful assistant.' },
#         { role: 'user', content: 'Who was the first president of the United States?' }
#     ],
#     max_tokens: 100
# }

# result = Net::HTTP.post URI.parse(SERVER + '/chat/completions'), JSON.generate(chatCompleteReq), 'Content-Type' => 'text/json'
# p result.each_header.to_h
# puts result.body

# verifyReq = {
#     request: chatCompleteReq,
#     response: JSON::parse(result.body)
# }

# puts "verifying..."
# result = Net::HTTP.post URI.parse(SERVER + '/chat/verify_completion'), JSON.generate(verifyReq), 'Content-Type' => 'text/json'
# p result.each_header.to_h
# puts result.body


