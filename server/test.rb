require 'net/http'

url = URI.parse 'http://localhost:7331/complete'

result = Net::HTTP.post url, 'The first man to', 'Content-Type' => 'test/plain'

p result.each_header.to_h
puts result.body
