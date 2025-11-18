"""
Generate Test Data for PII Verifier Experiments

Creates JSONL test files with various PII entity types for testing models.

Usage:
    python generate_test_data.py --output data/test_emails.jsonl --count 100 --type email
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


class TestDataGenerator:
    """Generate test data for PII verification experiments."""

    def __init__(self):
        # Sample data
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily",
            "Robert", "Lisa", "William", "Jennifer", "James", "Mary"
        ]

        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"
        ]

        self.domains = [
            "example.com", "test.com", "company.com", "email.com",
            "business.org", "service.net", "support.io"
        ]

        self.schools = [
            "Elementary School", "High School", "Middle School",
            "University", "College", "Academy", "Institute"
        ]

        self.buildings = [
            "Library", "Hospital", "Center", "Building", "Plaza",
            "Tower", "Hall", "Complex"
        ]

        self.streets = [
            "Main Street", "Oak Avenue", "Park Road", "First Street",
            "Washington Boulevard", "Elm Drive", "Maple Lane"
        ]

        self.crypto_prefixes = {
            "bitcoin": ["1", "3", "bc1"],
            "ethereum": ["0x"]
        }

        self.countries = ["DE", "FR", "GB", "IT", "ES", "NL", "BE", "CH"]

        self.medications = [
            "lisinopril", "metformin", "amlodipine", "metoprolol",
            "omeprazole", "simvastatin", "losartan", "albuterol"
        ]

    def generate_email_tests(self, count: int) -> List[Dict]:
        """Generate email test cases (true positives and false positives)."""
        tests = []

        for i in range(count):
            first = random.choice(self.first_names).lower()
            last = random.choice(self.last_names).lower()
            domain = random.choice(self.domains)
            email = f"{first}.{last}@{domain}"

            # 80% true positives, 20% false positives
            if random.random() < 0.8:
                # True positive - email in contact context
                contexts = [
                    f"Contact {email} for support",
                    f"Email us at {email}",
                    f"Reach out to {email} with questions",
                    f"Send your inquiry to {email}",
                    f"For assistance, email {email}"
                ]
                context = random.choice(contexts)
                is_true_positive = True
            else:
                # False positive - email-like pattern in other context
                fake_email = f"{first}.{last}@{domain}"
                contexts = [
                    f"The format is username@domain.com like {fake_email}",
                    f"Example: {fake_email} (not a real email)",
                    f"Template: name@company.com (e.g., {fake_email})"
                ]
                context = random.choice(contexts)
                email = fake_email
                is_true_positive = False

            tests.append({
                "recordId": f"email_{i:04d}",
                "input": context,
                "entityType": "EMAIL",
                "entityValue": email,
                "metadata": {
                    "source": "test_generator",
                    "is_true_positive": is_true_positive
                }
            })

        return tests

    def generate_person_tests(self, count: int) -> List[Dict]:
        """Generate person name test cases."""
        tests = []

        for i in range(count):
            first = random.choice(self.first_names)
            last = random.choice(self.last_names)
            full_name = f"{first} {last}"

            # 50% true positives, 50% false positives (schools/buildings)
            if random.random() < 0.5:
                # True positive - actual person
                contexts = [
                    f"Dr. {full_name} is the lead researcher",
                    f"Patient {full_name} was admitted yesterday",
                    f"Employee {full_name} submitted the report",
                    f"{full_name} authored this paper",
                    f"Mr. {full_name} will attend the meeting"
                ]
                context = random.choice(contexts)
                is_true_positive = True
            else:
                # False positive - school/building name
                suffix = random.choice(self.schools + self.buildings)
                school_name = f"{full_name} {suffix}"
                contexts = [
                    f"{school_name} is located in California",
                    f"Students at {school_name} performed well",
                    f"The {school_name} opened in 1985",
                    f"Welcome to {school_name}"
                ]
                context = random.choice(contexts)
                is_true_positive = False

            tests.append({
                "recordId": f"person_{i:04d}",
                "input": context,
                "entityType": "PERSON",
                "entityValue": full_name,
                "metadata": {
                    "source": "test_generator",
                    "is_true_positive": is_true_positive
                }
            })

        return tests

    def generate_financial_tests(self, count: int) -> List[Dict]:
        """Generate financial entity test cases."""
        tests = []

        for i in range(count):
            test_type = random.choice(["CRYPTO", "IBAN", "SWIFT_CODE"])

            if test_type == "CRYPTO":
                # Generate Bitcoin or Ethereum address
                crypto_type = random.choice(["bitcoin", "ethereum"])

                if crypto_type == "bitcoin":
                    prefix = random.choice(self.crypto_prefixes["bitcoin"])
                    address = prefix + ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=33))
                else:
                    address = "0x" + ''.join(random.choices('0123456789abcdef', k=40))

                contexts = [
                    f"Send Bitcoin to {address}",
                    f"Wallet address: {address}",
                    f"Crypto payment: {address}",
                    f"Deposit to {address}"
                ]
                context = random.choice(contexts)
                entity_value = address

            elif test_type == "IBAN":
                # Generate IBAN
                country = random.choice(self.countries)
                check_digits = random.randint(10, 99)
                bank_code = ''.join(random.choices('0123456789', k=8))
                account = ''.join(random.choices('0123456789', k=10))
                iban = f"{country}{check_digits}{bank_code}{account}"

                contexts = [
                    f"Bank account IBAN: {iban}",
                    f"Transfer to {iban}",
                    f"IBAN number: {iban}",
                    f"Account: {iban}"
                ]
                context = random.choice(contexts)
                entity_value = iban

            else:  # SWIFT_CODE
                # Generate SWIFT/BIC code
                bank = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                country = random.choice(self.countries)
                location = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=2))
                branch = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=3))
                swift = f"{bank}{country}{location}{branch}"

                contexts = [
                    f"SWIFT code: {swift}",
                    f"BIC: {swift}",
                    f"Bank identifier: {swift}",
                    f"Transfer via {swift}"
                ]
                context = random.choice(contexts)
                entity_value = swift

            tests.append({
                "recordId": f"financial_{i:04d}",
                "input": context,
                "entityType": test_type,
                "entityValue": entity_value,
                "metadata": {
                    "source": "test_generator",
                    "is_true_positive": True
                }
            })

        return tests

    def generate_medical_tests(self, count: int) -> List[Dict]:
        """Generate medical/PHI entity test cases."""
        tests = []

        for i in range(count):
            test_type = random.choice(["MEDICAL_RECORD_NUMBER", "PATIENT", "MEDICATION"])

            if test_type == "MEDICAL_RECORD_NUMBER":
                # Generate MRN
                mrn = f"MRN-{random.randint(100000, 999999)}"

                contexts = [
                    f"Patient {mrn} was admitted",
                    f"Medical record number: {mrn}",
                    f"Review {mrn} chart",
                    f"Discharge summary for {mrn}"
                ]
                context = random.choice(contexts)
                entity_value = mrn

            elif test_type == "PATIENT":
                # Generate patient name
                first = random.choice(self.first_names)
                last = random.choice(self.last_names)
                patient_name = f"{first} {last}"

                contexts = [
                    f"Patient {patient_name} presents with fever",
                    f"{patient_name} was diagnosed with diabetes",
                    f"Admitted patient {patient_name}",
                    f"Treatment plan for {patient_name}"
                ]
                context = random.choice(contexts)
                entity_value = patient_name

            else:  # MEDICATION
                # Generate medication
                medication = random.choice(self.medications)
                dosage = f"{random.choice([5, 10, 20, 40, 80])}mg"

                contexts = [
                    f"Prescribed {medication} {dosage} daily",
                    f"Patient takes {medication}",
                    f"Medication: {medication} {dosage}",
                    f"Continue {medication} treatment"
                ]
                context = random.choice(contexts)
                entity_value = medication

            tests.append({
                "recordId": f"medical_{i:04d}",
                "input": context,
                "entityType": test_type,
                "entityValue": entity_value,
                "metadata": {
                    "source": "test_generator",
                    "is_true_positive": True
                }
            })

        return tests


def main():
    parser = argparse.ArgumentParser(description="Generate test data for PII verifier experiments")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--count", type=int, default=100, help="Number of test cases to generate")
    parser.add_argument(
        "--type",
        choices=["email", "person", "financial", "medical", "all"],
        default="email",
        help="Type of test data to generate"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate test data
    generator = TestDataGenerator()

    if args.type == "email":
        tests = generator.generate_email_tests(args.count)
    elif args.type == "person":
        tests = generator.generate_person_tests(args.count)
    elif args.type == "financial":
        tests = generator.generate_financial_tests(args.count)
    elif args.type == "medical":
        tests = generator.generate_medical_tests(args.count)
    else:  # all
        count_per_type = args.count // 4
        tests = (
            generator.generate_email_tests(count_per_type) +
            generator.generate_person_tests(count_per_type) +
            generator.generate_financial_tests(count_per_type) +
            generator.generate_medical_tests(count_per_type)
        )

    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for test in tests:
            json_line = json.dumps(test, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Generated {len(tests)} test cases")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
