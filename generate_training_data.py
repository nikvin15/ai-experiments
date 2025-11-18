"""
Generate Training Data for Data Elements

Creates 504 test cases (126 data elements × 4 examples) for PII verification model training.
Covers all domains: Healthcare, Finance, E-commerce, HR, Education, Government, Social Media,
Technology, Marketing (HubSpot), Sales, Slack Messages, and Resume Data.

Usage:
    python generate_training_data.py --input-json /path/to/default_data_elements.json \
                                      --output data/training_all_elements.jsonl
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class DataElementExampleGenerator:
    """Generates realistic examples for each data element across different domains."""

    def __init__(self):
        # Domain templates for each data element category
        self.domain_templates = self._init_domain_templates()
        self.false_positive_templates = self._init_false_positive_templates()
        self.realistic_json_templates = self._init_realistic_json_templates()

    def _init_domain_templates(self) -> Dict:
        """Initialize domain-specific templates for generating examples."""
        return {
            # HEALTHCARE/MEDICAL DOMAIN
            "healthcare": {
                "Email Address": [
                    "Patient john.doe@healthclinic.com requires lab results",
                    "Dr. Sarah Smith can be reached at ssmith@hospital.org",
                    "Medical records for patient contact: jdoe@email.com"
                ],
                "Phone Number": [
                    "Emergency contact for patient: (555) 123-4567",
                    "Schedule appointment by calling 555-987-6543",
                    "Doctor's office number: +1-555-246-8000"
                ],
                "Social Security Number": [
                    "Patient SSN 123-45-6789 requires authorization for surgery",
                    "Insurance verification for SSN: 987-65-4321",
                    "Medicare claim SSN 456-78-9012 needs review"
                ],
                "Date of Birth": [
                    "Patient DOB: 05/15/1985 scheduled for consultation",
                    "Birth date 1990-03-22 verified for medical records",
                    "Patient birthdate 12/25/1978 on file"
                ],
            },
            # FINANCE/BANKING DOMAIN
            "finance": {
                "Bank Account Details": [
                    "Transfer funds to account number 1234567890123456",
                    "IBAN DE89370400440532013000 for international payment",
                    "Routing 123456789 and account 987654321012"
                ],
                "Card Number": [
                    "Credit card 4532-1234-5678-9010 charged for purchase",
                    "Debit card ending in 5678 declined transaction",
                    "Payment card 5425233430109903 on file"
                ],
                "Credit Score": [
                    "Applicant credit score 720 approved for loan",
                    "FICO score 680 requires additional documentation",
                    "Credit rating 750 qualifies for premium rate"
                ],
            },
            # E-COMMERCE/RETAIL DOMAIN
            "ecommerce": {
                "Email Address": [
                    "Order confirmation sent to customer@shop.com",
                    "Shipping notification for jsmith@retail.net",
                    "Account registered with email: buyer123@gmail.com"
                ],
                "Address": [
                    "Delivery address: 123 Main St, Springfield, IL 62701",
                    "Billing addr: 456 Oak Avenue, Apt 2B, New York, NY 10001",
                    "Ship to: 789 Pine Road, Seattle, WA 98101"
                ],
                "Order Details": [
                    "Order #ORD-2024-123456 total $299.99 shipped",
                    "Purchase transaction ID TXN-789012 processed successfully",
                    "Invoice INV-2024-001 for items: laptop, mouse, keyboard"
                ],
            },
            # HR/EMPLOYMENT DOMAIN
            "hr": {
                "Employee Code": [
                    "Employee ID EMP-123456 assigned to John Smith",
                    "Staff number STAFF-789012 for payroll processing",
                    "Worker code WRK-345678 clocked in at 9:00 AM"
                ],
                "Salary": [
                    "Annual salary $85,000 effective January 2024",
                    "Monthly compensation $7,500 plus benefits",
                    "Base pay $65,000 with performance bonus"
                ],
                "Past Employers": [
                    "Previous employment at Google Inc (2018-2022)",
                    "Work history: Microsoft Corp, Amazon AWS, IBM",
                    "Former employer: Acme Corporation, 2015-2020"
                ],
            },
            # EDUCATION/GOVERNMENT DOMAIN
            "education": {
                "First Name": [
                    "Student first name: Emily registered for semester",
                    "Given name Michael on transcript records",
                    "Applicant fname: Sarah for university admission"
                ],
                "Educational History": [
                    "Bachelor's degree from Stanford University 2018",
                    "Master's program at MIT, graduated 2020",
                    "PhD from Harvard, Computer Science, 2022"
                ],
                "Academic Transcripts": [
                    "Transcript shows GPA 3.8, Dean's List all semesters",
                    "Diploma certificate from UCLA dated June 2019",
                    "Degree verification: BS in Engineering, honors"
                ],
            },
            # SOCIAL MEDIA/TECHNOLOGY DOMAIN
            "technology": {
                "IP Address": [
                    "User logged in from IP 203.0.113.45",
                    "Request originated from 192.0.2.100",
                    "Client IP address: 2001:db8::1 geo-located in EU"
                ],
                "Device Identifier": [
                    "Device IMEI 123456789012345 registered to account",
                    "Mobile device ID: android-abc123def456 active",
                    "iOS device token: ios-789ghi012jkl345 for push"
                ],
                "Cookies": [
                    "Session cookie _ga=GA1.2.123456789.1234567890 set",
                    "Tracking cookie fbp=fb.1.1234567890.987654321 found",
                    "User preference cookies: theme=dark, lang=en"
                ],
            },
        }

    def _init_false_positive_templates(self) -> Dict:
        """Initialize false positive patterns that look like PII but aren't."""
        return {
            "Email Address": [
                "System error: invalid format username@domain.com (example only)",
                "Email syntax: noreply@example.com is not a valid user address",
                "Documentation shows format: user@example.org for reference"
            ],
            "Phone Number": [
                "For support, call toll-free 1-800-HELP-NOW (1-800-435-7669)",
                "Emergency services dial 911 in USA or 112 in Europe",
                "Fax number 555-0100 is the company general line"
            ],
            "Social Security Number": [
                "Bug ticket ID SSN-123-456 assigned to security team",
                "Pattern format ###-##-#### where # is any digit",
                "System identifier SSN-TEST-789 for staging environment"
            ],
            "First Name": [
                "Product name SmartPhone Pro launched today",
                "Building name Johnson Hall will be renovated",
                "School name Washington Elementary opens Monday"
            ],
            "Date of Birth": [
                "Company founded on 01/15/1995 celebrates anniversary",
                "Product release date 2024-03-15 confirmed by team",
                "Service launch birthdate: 12/25/2020"
            ],
            "Credit Score": [
                "Test score 720 out of 800 on certification exam",
                "Game score 680 points achieved in level 5",
                "Quality score 750 meets minimum threshold"
            ],
            "IP Address": [
                "Server IP 127.0.0.1 is localhost for testing",
                "Internal network 10.0.0.1 gateway configuration",
                "Private subnet 192.168.1.1 for local development"
            ],
            "Employee Code": [
                "Product code EMP-SKU-789 for inventory system",
                "Error code EMP-500 indicates server timeout",
                "Status code STAFF-ACTIVE for application state"
            ],
        }

    def _init_realistic_json_templates(self) -> Dict:
        """Initialize realistic JSON templates for various domains."""
        return {
            "user_profile": {
                "template": {
                    "userId": "USR-{id}",
                    "firstName": "{firstName}",
                    "lastName": "{lastName}",
                    "email": "{email}",
                    "phone": "{phone}",
                    "dateOfBirth": "{dob}",
                    "gender": "{gender}",
                    "address": {
                        "street": "{street}",
                        "city": "{city}",
                        "state": "{state}",
                        "zipCode": "{zip}"
                    },
                    "accountStatus": "active",
                    "memberSince": "2022-03-15"
                },
                "pii_fields": {
                    "firstName": "First Name",
                    "lastName": "Last Name",
                    "email": "Email Address",
                    "phone": "Phone Number",
                    "dateOfBirth": "Date of Birth",
                    "gender": "Gender",
                    "address.street": "Address",
                    "address.city": "Address",
                    "address.state": "Address",
                    "address.zipCode": "Address"
                }
            },
            "employee_record": {
                "template": {
                    "employeeId": "EMP-{empId}",
                    "personalInfo": {
                        "fullName": "{fullName}",
                        "ssn": "{ssn}",
                        "dob": "{dob}",
                        "email": "{email}",
                        "phone": "{phone}"
                    },
                    "employment": {
                        "department": "{department}",
                        "jobTitle": "{jobTitle}",
                        "salary": "{salary}",
                        "hireDate": "{hireDate}"
                    },
                    "payroll": {
                        "bankAccount": "{bankAccount}",
                        "routingNumber": "{routing}"
                    }
                },
                "pii_fields": {
                    "personalInfo.fullName": "First Name",  # Can map to multiple
                    "personalInfo.ssn": "Social Security Number",
                    "personalInfo.dob": "Date of Birth",
                    "personalInfo.email": "Email Address",
                    "personalInfo.phone": "Phone Number",
                    "employment.salary": "Salary",
                    "payroll.bankAccount": "Bank Account Details",
                    "employeeId": "Employee Code"
                }
            },
            "ecommerce_order": {
                "template": {
                    "orderId": "ORD-{orderId}",
                    "customer": {
                        "name": "{customerName}",
                        "email": "{email}",
                        "phone": "{phone}"
                    },
                    "shipping": {
                        "address": "{shippingAddress}",
                        "city": "{city}",
                        "zipCode": "{zip}"
                    },
                    "payment": {
                        "method": "credit_card",
                        "cardNumber": "{cardNumber}",
                        "cardHolder": "{cardHolder}"
                    },
                    "items": [
                        {"sku": "PROD-001", "quantity": 2, "price": 29.99}
                    ],
                    "total": 59.98
                },
                "pii_fields": {
                    "customer.name": "First Name",
                    "customer.email": "Email Address",
                    "customer.phone": "Phone Number",
                    "shipping.address": "Address",
                    "payment.cardNumber": "Card Number"
                }
            },
            "hubspot_contact": {
                "template": {
                    "vid": "{contactId}",
                    "properties": {
                        "firstname": {"value": "{firstName}"},
                        "lastname": {"value": "{lastName}"},
                        "email": {"value": "{email}"},
                        "phone": {"value": "{phone}"},
                        "company": {"value": "{company}"},
                        "jobtitle": {"value": "{jobTitle}"},
                        "address": {"value": "{address}"},
                        "city": {"value": "{city}"},
                        "state": {"value": "{state}"}
                    },
                    "identity-profiles": [
                        {
                            "vid": "{contactId}",
                            "identities": [
                                {"type": "EMAIL", "value": "{email}"}
                            ]
                        }
                    ]
                },
                "pii_fields": {
                    "properties.firstname.value": "First Name",
                    "properties.lastname.value": "Last Name",
                    "properties.email.value": "Email Address",
                    "properties.phone.value": "Phone Number",
                    "properties.address.value": "Address",
                    "properties.company.value": "Past Employers"
                }
            },
            "slack_message": {
                "template": {
                    "type": "message",
                    "user": "U{userId}",
                    "text": "{messageText}",
                    "ts": "1234567890.123456",
                    "channel": "C{channelId}",
                    "user_profile": {
                        "real_name": "{realName}",
                        "email": "{email}",
                        "phone": "{phone}",
                        "title": "{title}"
                    }
                },
                "pii_fields": {
                    "user_profile.real_name": "First Name",
                    "user_profile.email": "Email Address",
                    "user_profile.phone": "Phone Number"
                }
            },
            "resume_json": {
                "template": {
                    "candidate": {
                        "name": "{fullName}",
                        "email": "{email}",
                        "phone": "{phone}",
                        "location": "{city}, {state}"
                    },
                    "experience": [
                        {
                            "company": "{company1}",
                            "title": "{title1}",
                            "duration": "2018-2022"
                        },
                        {
                            "company": "{company2}",
                            "title": "{title2}",
                            "duration": "2015-2018"
                        }
                    ],
                    "education": [
                        {
                            "degree": "{degree}",
                            "school": "{school}",
                            "year": 2015
                        }
                    ],
                    "skills": ["Python", "JavaScript", "SQL"]
                },
                "pii_fields": {
                    "candidate.name": "First Name",
                    "candidate.email": "Email Address",
                    "candidate.phone": "Phone Number",
                    "experience": "Past Employers",
                    "education": "Educational History"
                }
            },
            "healthcare_record": {
                "template": {
                    "patientId": "PAT-{patientId}",
                    "demographics": {
                        "name": "{fullName}",
                        "dob": "{dob}",
                        "ssn": "{ssn}",
                        "gender": "{gender}",
                        "contact": {
                            "phone": "{phone}",
                            "email": "{email}",
                            "address": "{address}"
                        }
                    },
                    "insurance": {
                        "provider": "{insuranceProvider}",
                        "policyNumber": "{policyNumber}",
                        "groupNumber": "{groupNumber}"
                    },
                    "medical": {
                        "conditions": ["{condition}"],
                        "medications": ["{medication}"],
                        "allergies": ["{allergy}"]
                    }
                },
                "pii_fields": {
                    "demographics.name": "First Name",
                    "demographics.dob": "Date of Birth",
                    "demographics.ssn": "Social Security Number",
                    "demographics.gender": "Gender",
                    "demographics.contact.phone": "Phone Number",
                    "demographics.contact.email": "Email Address",
                    "demographics.contact.address": "Address",
                    "insurance.policyNumber": "Insurance Number",
                    "medical.conditions": "Illness or Medical Condition"
                }
            },
            "financial_account": {
                "template": {
                    "accountId": "ACC-{accountId}",
                    "accountHolder": {
                        "name": "{fullName}",
                        "ssn": "{ssn}",
                        "dob": "{dob}",
                        "email": "{email}",
                        "phone": "{phone}"
                    },
                    "account": {
                        "type": "checking",
                        "number": "{accountNumber}",
                        "routingNumber": "{routing}",
                        "balance": "{balance}"
                    },
                    "creditInfo": {
                        "score": "{creditScore}",
                        "cards": [
                            {
                                "type": "credit",
                                "number": "{cardNumber}",
                                "expiry": "{expiry}"
                            }
                        ]
                    }
                },
                "pii_fields": {
                    "accountHolder.name": "First Name",
                    "accountHolder.ssn": "Social Security Number",
                    "accountHolder.dob": "Date of Birth",
                    "accountHolder.email": "Email Address",
                    "accountHolder.phone": "Phone Number",
                    "account.number": "Bank Account Details",
                    "creditInfo.score": "Credit Score",
                    "creditInfo.cards": "Card Number"
                }
            }
        }

    def generate_healthcare_example(self, element: Dict, is_true_positive: bool) -> Dict:
        """Generate healthcare domain example."""
        name = element["name"]

        if not is_true_positive and name in self.false_positive_templates:
            text = random.choice(self.false_positive_templates[name])
            return {
                "input": text,
                "PIIs": [],
                "domain": "healthcare",
                "is_true_positive": False,
                "reason": "False positive pattern in healthcare context"
            }

        # Generate realistic healthcare examples
        examples = {
            "Criminal Records History": "Background check flagged criminal history for healthcare worker application",
            "Email Address": "Patient mary.johnson@healthmail.com scheduled for follow-up",
            "Phone Number": "Patient contact number (555) 234-5678 for appointment reminders",
            "Address": "Patient residence: 789 Wellness Drive, Boston, MA 02108",
            "Emergency Contact Details": "Emergency contact: spouse at 555-345-6789, relation: husband",
            "Social Security Number": "Insurance claim for SSN 234-56-7890 submitted to Medicare",
            "Date of Birth": "Patient DOB 08/12/1972 confirmed for prescription refill",
            "First Name": "Patient first name Robert registered in EMR system",
            "Last Name": "Last name Anderson on medical chart requires update",
            "Gender": "Patient gender: female for gynecology appointment",
            "Disability or Specific Condition": "Patient has type 2 diabetes requiring insulin therapy",
            "Illness or Medical Condition": "Medical condition: hypertension controlled with medication",
            "Medical Certificates": "Blood test report shows cholesterol level 200 mg/dL",
            "Insurance Number": "Health insurance policy number INS-987654 active until 2025",
            "Weight": "Patient weight 175 lbs recorded at checkup",
            "Height": "Patient height 5'9\" measured during physical exam",
        }

        text = examples.get(name, f"Healthcare data for {name.lower()}: sample value in medical record")

        # Extract PII from text
        pii_info = self._extract_pii_from_text(text, element)

        return {
            "input": text,
            "PIIs": pii_info,
            "domain": "healthcare",
            "is_true_positive": True
        }

    def generate_finance_example(self, element: Dict, is_true_positive: bool) -> Dict:
        """Generate finance domain example."""
        name = element["name"]

        if not is_true_positive and name in self.false_positive_templates:
            text = random.choice(self.false_positive_templates[name])
            return {
                "input": text,
                "PIIs": [],
                "domain": "finance",
                "is_true_positive": False,
                "reason": "False positive pattern in financial context"
            }

        examples = {
            "Bank Account Details": "Wire transfer to account 9876543210987654, routing 021000021",
            "Card Number": "Credit card 4111111111111111 charged $150.00 for transaction",
            "Card Issuer": "Card issued by Chase Bank, Visa network",
            "Credit Score": "Customer FICO score 780 qualifies for premium loan terms",
            "Tax Information": "Tax ID 12-3456789 for business account setup",
            "Salary": "Annual income $95,000 verified for mortgage application",
            "Retirement Account Information": "401k account balance $250,000 as of Q4 2024",
            "Stock or Equity Grants": "Employee stock options: 1000 shares vested",
            "Insurance Number": "Policy number POL-123456789 for auto insurance claim",
            "VPA Address": "UPI payment to user@okaxis for bill settlement",
            "Payment Mode": "Payment method selected: credit card ending 4567",
            "Payroll Information": "Payroll processed for pay period ending 12/31/2024",
            "Expenses": "Travel expense reimbursement $450.00 approved",
            "Benefits": "Health insurance premium $500/month employer-paid",
            "Email Address": "Account statement sent to customer@bank.com",
            "Phone Number": "Customer service call from (555) 456-7890 regarding loan",
            "Address": "Billing address: 321 Finance St, Chicago, IL 60601",
            "First Name": "Account holder first name: Jennifer on file",
            "Last Name": "Surname: Williams for credit card application",
            "Date of Birth": "DOB 04/20/1985 verified for identity confirmation",
            "Social Security Number": "SSN 345-67-8901 for tax reporting purposes",
        }

        text = examples.get(name, f"Financial record contains {name.lower()} for customer account")
        pii_info = self._extract_pii_from_text(text, element)

        return {
            "input": text,
            "PIIs": pii_info,
            "domain": "finance",
            "is_true_positive": True
        }

    def generate_json_example(self, element: Dict, is_true_positive: bool) -> Dict:
        """Generate realistic JSON format example using templates."""
        name = element["name"]

        if not is_true_positive:
            # JSON false positives - system/config data that looks like PII patterns
            json_false_positives = {
                "Email Address": '{"notification_email": "noreply@system.com", "type": "automated", "enabled": true}',
                "Phone Number": '{"support_line": "1-800-SUPPORT", "type": "tollfree", "hours": "24/7"}',
                "Social Security Number": '{"error_code": "SSN-ERR-500", "message": "validation failed", "retry": true}',
                "Employee Code": '{"product_sku": "EMP-ITEM-789", "category": "electronics", "stock": 50}',
                "IP Address": '{"server_ip": "127.0.0.1", "environment": "localhost", "status": "testing"}',
                "First Name": '{"product_name": "SmartPhone", "brand": "TechCorp", "model": "Pro"}',
            }

            json_text = json_false_positives.get(name, '{"config": "test_mode", "status": "active", "env": "development"}')

            return {
                "input": json_text,
                "PIIs": [],
                "domain": "technology",
                "is_true_positive": False,
                "reason": "JSON contains non-PII data that matches pattern"
            }

        # Select a realistic JSON template based on data element category
        template_choice = self._select_json_template_for_element(element)
        json_text = self._generate_realistic_json(template_choice, element)

        # Extract all PIIs from the JSON (may contain multiple types)
        pii_list = self._extract_piis_from_realistic_json(json_text, template_choice)

        return {
            "input": json_text,
            "PIIs": pii_list,
            "domain": self._get_domain_for_template(template_choice),
            "is_true_positive": True
        }

    def _select_json_template_for_element(self, element: Dict) -> str:
        """Select appropriate JSON template based on element category/name."""
        category = element["category"]
        name = element["name"]

        # Map categories to templates
        if category in ["Health Data", "Biometric Data"] or "Medical" in name or "Patient" in name:
            return "healthcare_record"
        elif category in ["Financial Data", "Purchase Data"] or name in ["Bank Account Details", "Card Number", "Credit Score"]:
            return "financial_account"
        elif category == "Professional & Employment Background" or name in ["Past Employers", "Salary"]:
            return random.choice(["employee_record", "resume_json"])
        elif "HubSpot" in name or "Lead" in name:
            return "hubspot_contact"
        elif category in ["Contact Data", "Personal Identification"]:
            return random.choice(["user_profile", "ecommerce_order"])
        elif name == "Employee Code":
            return "employee_record"
        else:
            # Default: use user_profile for most general PII
            return "user_profile"

    def _generate_realistic_json(self, template_name: str, element: Dict) -> str:
        """Generate a realistic JSON string from a template."""
        template_data = self.realistic_json_templates[template_name]
        template = template_data["template"]

        # Sample realistic values
        sample_data = {
            "id": str(random.randint(10000, 99999)),
            "firstName": random.choice(["Sarah", "Michael", "Emily", "David", "Jennifer"]),
            "lastName": random.choice(["Williams", "Johnson", "Smith", "Brown", "Davis"]),
            "fullName": "Sarah Williams",
            "email": "sarah.williams@company.com",
            "phone": "555-234-5678",
            "dob": "1985-06-15",
            "gender": random.choice(["female", "male", "non-binary"]),
            "street": "123 Main Street",
            "city": random.choice(["San Francisco", "New York", "Boston", "Seattle"]),
            "state": random.choice(["CA", "NY", "MA", "WA"]),
            "zip": random.choice(["94102", "10001", "02108", "98101"]),
            "ssn": "123-45-6789",
            "empId": str(random.randint(1000, 9999)),
            "department": random.choice(["Engineering", "Sales", "Marketing", "HR"]),
            "jobTitle": random.choice(["Software Engineer", "Sales Manager", "Product Manager"]),
            "salary": random.randint(60000, 150000),
            "hireDate": "2020-01-15",
            "bankAccount": "1234567890123456",
            "routing": "021000021",
            "orderId": str(random.randint(100000, 999999)),
            "customerName": "John Doe",
            "shippingAddress": "456 Oak Avenue",
            "cardNumber": "4111111111111111",
            "cardHolder": "JOHN DOE",
            "contactId": random.randint(10000, 99999),
            "company": random.choice(["Google", "Microsoft", "Amazon", "Apple"]),
            "userId": str(random.randint(1000, 9999)),
            "channelId": str(random.randint(100, 999)),
            "messageText": "Let's schedule a meeting for next week",
            "realName": "Sarah Williams",
            "title": "Senior Engineer",
            "company1": "Google",
            "company2": "Microsoft",
            "title1": "Software Engineer",
            "title2": "Junior Developer",
            "degree": "Bachelor of Science in Computer Science",
            "school": "Stanford University",
            "patientId": str(random.randint(10000, 99999)),
            "insuranceProvider": "Blue Cross",
            "policyNumber": "POL-" + str(random.randint(100000, 999999)),
            "groupNumber": "GRP-" + str(random.randint(1000, 9999)),
            "condition": "Type 2 Diabetes",
            "medication": "Metformin",
            "allergy": "Penicillin",
            "accountId": str(random.randint(10000, 99999)),
            "accountNumber": "9876543210987654",
            "balance": random.randint(1000, 50000),
            "creditScore": random.randint(650, 850),
            "expiry": "12/25"
        }

        # Recursively populate template
        def populate(obj):
            if isinstance(obj, dict):
                return {k: populate(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [populate(item) for item in obj]
            elif isinstance(obj, str):
                # Replace placeholders
                result = obj
                for key, value in sample_data.items():
                    result = result.replace(f"{{{key}}}", str(value))
                return result
            else:
                return obj

        populated = populate(template)
        return json.dumps(populated, indent=None, ensure_ascii=False)

    def _extract_piis_from_realistic_json(self, json_text: str, template_name: str) -> List[str]:
        """Extract all PII types from a realistic JSON using template mapping."""
        template_data = self.realistic_json_templates[template_name]
        pii_fields = template_data["pii_fields"]

        try:
            data = json.loads(json_text)
            found_piis = set()  # Use set to avoid duplicates

            def search_dict(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key

                        # Check if this key maps to a known PII type
                        if full_key in pii_fields:
                            found_piis.add(pii_fields[full_key])

                        # Recurse into nested objects
                        if isinstance(value, dict):
                            search_dict(value, full_key)
                        elif isinstance(value, list) and value and isinstance(value[0], dict):
                            # For arrays of objects, check the array itself
                            if full_key in pii_fields:
                                found_piis.add(pii_fields[full_key])

            search_dict(data)
            return sorted(list(found_piis))  # Return sorted list for consistency
        except:
            # Fallback: return empty list for false positives
            return []

    def _get_domain_for_template(self, template_name: str) -> str:
        """Get domain name for a template."""
        domain_map = {
            "user_profile": "technology",
            "employee_record": "hr",
            "ecommerce_order": "ecommerce",
            "hubspot_contact": "marketing",
            "slack_message": "slack",
            "resume_json": "resume",
            "healthcare_record": "healthcare",
            "financial_account": "finance"
        }
        return domain_map.get(template_name, "technology")

    def _extract_pii_from_text(self, text: str, element: Dict) -> List[str]:
        """Extract PII information from text based on element type.

        Returns list of data element names (strings) found in text.
        """
        name = element["name"]

        # Simple extraction - just return the data element name if it should be present
        # For false positives, this will return empty list
        return [name] if text else []

    def _extract_pii_from_json(self, json_text: str, element: Dict) -> List[str]:
        """Extract PII information from JSON string.

        Returns list of data element names (strings) found in JSON.
        """
        # For now, just return the element name
        # This will be updated when we add realistic JSON templates
        name = element["name"]
        return [name] if json_text else []

    def _extract_all_piis_from_json(self, json_text: str, data_elements_map: Dict) -> List[str]:
        """Extract all PII types from a realistic JSON object.

        Args:
            json_text: JSON string to analyze
            data_elements_map: Map of field names to data element names

        Returns:
            List of data element names found in the JSON
        """
        try:
            data = json.loads(json_text)
            found_piis = []

            # Recursively search for known PII fields
            def search_dict(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        # Check if this key maps to a known PII type
                        if full_key in data_elements_map:
                            found_piis.append(data_elements_map[full_key])
                        elif key in data_elements_map:
                            found_piis.append(data_elements_map[key])
                        # Recurse into nested objects
                        if isinstance(value, dict):
                            search_dict(value, full_key)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    search_dict(item, full_key)

            search_dict(data)
            return found_piis
        except:
            return []


class TrainingDataGenerator:
    """Main generator for creating training dataset."""

    def __init__(self, data_elements_path: str):
        self.data_elements = self._load_data_elements(data_elements_path)
        self.example_generator = DataElementExampleGenerator()
        self.false_positive_count = 0
        self.true_positive_count = 0

    def _load_data_elements(self, path: str) -> List[Dict]:
        """Load data elements from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_training_set(self) -> List[Dict]:
        """Generate complete training set with 504 examples."""
        training_data = []
        record_id = 1

        # Calculate false positive distribution (20% of 504 = ~101)
        total_examples = len(self.data_elements) * 4  # 126 * 4 = 504
        target_false_positives = int(total_examples * 0.20)  # ~101

        # Distribute false positives evenly across examples
        # Each element gets 4 examples, so ~0.8 false positive per element
        # We'll randomize which elements get false positives

        print(f"Generating training data for {len(self.data_elements)} data elements...")
        print(f"Total examples: {total_examples}")
        print(f"Target false positives: {target_false_positives}")

        for idx, element in enumerate(self.data_elements, 1):
            # Generate 4 examples per element
            examples = self._generate_examples_for_element(element, record_id)
            training_data.extend(examples)
            record_id += len(examples)

            if idx % 10 == 0:
                print(f"Progress: {idx}/{len(self.data_elements)} elements processed...")

        print(f"\nGeneration complete!")
        print(f"True positives: {self.true_positive_count}")
        print(f"False positives: {self.false_positive_count}")
        print(f"Total: {len(training_data)}")

        return training_data

    def _generate_examples_for_element(self, element: Dict, start_id: int) -> List[Dict]:
        """Generate 4 examples for a single data element (3 text + 1 JSON)."""
        examples = []

        # Determine if this element should have false positives (20% chance per example)
        false_positive_chances = [random.random() < 0.20 for _ in range(4)]

        # Example 1: Healthcare domain (text)
        is_fp = false_positive_chances[0]
        ex1 = self.example_generator.generate_healthcare_example(element, not is_fp)
        examples.append(self._create_training_record(
            record_id=f"de_{start_id:04d}_01",
            data_element=element,
            example_data=ex1
        ))
        if is_fp:
            self.false_positive_count += 1
        else:
            self.true_positive_count += 1

        # Example 2: Finance domain (text)
        is_fp = false_positive_chances[1]
        ex2 = self.example_generator.generate_finance_example(element, not is_fp)
        examples.append(self._create_training_record(
            record_id=f"de_{start_id:04d}_02",
            data_element=element,
            example_data=ex2
        ))
        if is_fp:
            self.false_positive_count += 1
        else:
            self.true_positive_count += 1

        # Example 3: E-commerce/HR domain (text) - reuse one of the generators
        is_fp = false_positive_chances[2]
        domain_choice = random.choice(["healthcare", "finance"])
        if domain_choice == "healthcare":
            ex3 = self.example_generator.generate_healthcare_example(element, not is_fp)
        else:
            ex3 = self.example_generator.generate_finance_example(element, not is_fp)
        ex3["domain"] = random.choice(["ecommerce", "hr", "education", "government"])
        examples.append(self._create_training_record(
            record_id=f"de_{start_id:04d}_03",
            data_element=element,
            example_data=ex3
        ))
        if is_fp:
            self.false_positive_count += 1
        else:
            self.true_positive_count += 1

        # Example 4: JSON format (technology/marketing/slack)
        is_fp = false_positive_chances[3]
        ex4 = self.example_generator.generate_json_example(element, not is_fp)
        examples.append(self._create_training_record(
            record_id=f"de_{start_id:04d}_04",
            data_element=element,
            example_data=ex4
        ))
        if is_fp:
            self.false_positive_count += 1
        else:
            self.true_positive_count += 1

        return examples

    def _create_training_record(self, record_id: str, data_element: Dict, example_data: Dict) -> Dict:
        """Create a simplified training record with only 3 keys: recordId, input, PIIs.

        Args:
            record_id: Unique identifier for the record
            data_element: Original data element definition
            example_data: Generated example with input, PIIs, domain, is_true_positive

        Returns:
            Dict with recordId, input, and PIIs (array of strings)
        """
        return {
            "recordId": record_id,
            "input": example_data["input"],
            "PIIs": example_data["PIIs"]  # Already a list of strings
        }


def main():
    parser = argparse.ArgumentParser(description="Generate training data for PII verification models")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to default_data_elements.json file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for training JSONL file"
    )

    args = parser.parse_args()

    # Generate training data
    generator = TrainingDataGenerator(args.input_json)
    training_data = generator.generate_training_set()

    # Write to JSONL file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(training_data)} records to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in training_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Training data generated successfully!")
    print(f"   File: {output_path}")
    print(f"   Records: {len(training_data)}")
    print(f"   True Positives: {generator.true_positive_count}")
    print(f"   False Positives: {generator.false_positive_count}")


if __name__ == "__main__":
    main()
