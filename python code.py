   def generate_primes(limit):
    """Generate a list of prime numbers up to a given limit."""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes

# Example usage:
limit = 50
prime_list = generate_primes(limit)
print(f"Prime numbers up to {limit}: {prime_list}")

                     