import qrcode

# The data you want to encode
data = "https://aburaksarac.github.io/AffectiveGameDesign/Website"

# Generate the QR code
img = qrcode.make(data)

# Save it locally
img.save("game_design_qr.svg")