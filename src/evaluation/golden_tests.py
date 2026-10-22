golden_tests = [
    {
        "question": "Can I claim 4200 hotel bill in Bangalore?",
        "expected_docs": ["AceCorp_Travel_Policy_v2.pdf", "AceCorp_Expense_Reimbursement_SOP.pdf"],
        "ground_truth": "Yes, if it is official travel and the hotel bill is within the metro city hotel limit with valid GST invoice and required approval."
    },
    {
        "question": "What approval is needed for software purchase above 50000?",
        "expected_docs": ["AceCorp_Procurement_SOP.pdf"],
        "ground_truth": "Approval depends on the amount: INR 50001 to INR 100000 requires Department Head approval; INR 100001 to INR 500000 requires Department Head and Procurement Head approval; above INR 500000 requires Business Unit Head and Finance Head approval. Software purchase also requires IT security review before PO creation."
    },
    {
        "question": "Can I deploy to production on Friday?",
        "expected_docs": ["AceCorp_Production_Deployment_SOP.pdf"],
        "ground_truth": "Regular production deployment on Friday is not allowed unless it is an emergency release with required approval."
    },
    {
        "question": "Can I claim cab fare after 10 PM?",
        "expected_docs": ["AceCorp_Travel_Policy_v2.pdf", "AceCorp_Expense_Reimbursement_SOP.pdf"],
        "ground_truth": "Cab fare after 10 PM is reimbursable if the ride invoice and required approval are provided."
    },
    {
        "question": "Can I take leave during notice period?",
        "expected_docs": ["AceCorp_Leave_Policy_2025.pdf"],
        "ground_truth": "Leave during notice period depends on the leave policy and requires the applicable approval."
    }
]
