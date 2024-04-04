import re

def validate_license_plate(plate):
    """
    Validates a license plate using a regular expression pattern.
    
    The pattern is "^[A-Z]+[0-9]+$". It checks if the input is a string that starts with one or more
    uppercase letters followed by one or more digits and ends with the end of the string.
    """
    pattern = r'^[A-Z]+[0-9]+$'
    
    # Check if the plate matches the pattern
    if re.match(pattern, plate):
        return True
    else:
        return False

# Example usage
# print(validate_license_plate("ABCDE12345")) # Should return True
# print(validate_license_plate("abcde12345")) # Should return False (lowercase letters)
# print(validate_license_plate("ABCDE"))       # Should return False (no digits)
# print(validate_license_plate("12345ABCDE")) # Should return False (starts with digits)
# print(validate_license_plate("ABCDE12345!")) # Should return False (contains special characters)