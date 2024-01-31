from pydicom import dcmread
from pynetdicom import AE
from pynetdicom.sop_class import Verification

# Initialize the AE
ae = AE()

# Add a verification presentation context
ae.add_requested_context(Verification)

# Define the peer AE details
scp_ip = "localhost"  # Replace with your SCP's IP
scp_port = 11112  # Replace with your SCP's Port

# Use the AE to perform a C-ECHO to the peer SCP
assoc = ae.associate(scp_ip, scp_port)
if assoc.is_established:
    # Perform a C-ECHO
    status = assoc.send_c_echo()

    # Check if the C-ECHO was successful
    if status:
        print("C-ECHO succeeded.")
    else:
        print("C-ECHO failed.")

    # Release the association
    assoc.release()
else:
    print("Association with SCP failed.")
