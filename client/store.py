from pydicom import dcmread
from pynetdicom import AE
from pynetdicom.sop_class import \
    DigitalMammographyXRayImageStorageForPresentation

# Initialize the AE
ae = AE()

# Add a presentation context
ae.add_requested_context(DigitalMammographyXRayImageStorageForPresentation)

# Read the DICOM file you want to send
file_path = r"C:\Users\Korisnik\Documents\GitHub\mammography\data\IORS\DISK 1\DICOM\ST000000\SE000000\MG000000"
ds = dcmread(file_path)

# Define the peer AE details
scp_ip = "localhost"  # Replace with your SCP's IP
scp_port = 11112  # Replace with your SCP's Port

# Use the AE to send the dataset to the peer SCP
assoc = ae.associate(scp_ip, scp_port)
if assoc.is_established:
    # Send the DICOM file
    status = assoc.send_c_store(ds)

    # Check if the file was sent successfully
    if status:
        print("DICOM file sent successfully.")
    else:
        print("Error sending DICOM file.")

    # Release the association
    assoc.release()
else:
    print("Association with SCP failed.")
