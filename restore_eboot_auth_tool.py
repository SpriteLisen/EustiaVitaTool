def eboot_bin_auth(eboot_file, key):
    fp_s = open(eboot_file, 'rb')
    data = bytearray(fp_s.read())
    fp_s.close()
    hex_ste = "".join(f"{b:02X}" for b in data[0x80:0x88])
    print(f"eboot 当前签名 => {hex_ste}")
    data[0x80:0x88] = key

    fp_s = open(eboot_file, 'wb')
    fp_s.write(data)
    fp_s.close()


if __name__ == "__main__":
    # 02000000 0000002F
    # 5201CE1C 10000021
    eboot_file_path = "eboot/device/eboot_patched.bin"  # eboot.bin 文件
    auth_file_path = "eboot/device/self_auth.bin"  # 签名文件

    auth_file = open(auth_file_path, "rb")
    auth_key = auth_file.read(8)
    hex_str = "".join(f"{b:02X}" for b in auth_key)
    auth_file.close()

    print(f"开始修改 {eboot_file_path} 文件签名")

    print(f"签名数据 => {hex_str}")

    eboot_bin_auth(
        eboot_file=eboot_file_path,
        key=auth_key
    )

    print("✅ 修改完毕")
