# RC Fragility Demo — Server Batches 04–08

แพ็กเกจพร้อมรันสำหรับสร้างข้อมูลอาคาร Batch 004–008 (ลำดับที่ 151–375) บน Windows โดยมีระบบ checkpoint/resume, หน้าจอติดตามสถานะ และชุด dependency สำหรับติดตั้งแบบ offline

## เริ่มใช้งาน

1. กด **Code → Download ZIP** หรือ clone repository นี้ไปยังไดรฟ์ที่มีพื้นที่เพียงพอ
2. แตก ZIP ให้เรียบร้อย (ห้ามรันจากในไฟล์ ZIP)
3. ต้องมี Windows 64-bit และ Python 3.10 64-bit
4. ดับเบิลคลิก `START_UI.bat`
5. เปิดหน้า Dashboard ตาม URL ที่โปรแกรมแสดง แล้วกดเริ่ม Batch

โปรแกรมบันทึกผลทันทีหลังงานย่อยสำเร็จ หากเครื่องดับหรือโปรแกรมหยุด ให้เปิด `START_UI.bat` อีกครั้ง ระบบจะทำต่อจาก checkpoint เดิมโดยไม่เริ่มใหม่ทั้งหมด

## ตำแหน่งผลลัพธ์

ไฟล์พร้อมนำกลับไปรวมกับเครื่องหลักอยู่ที่:

```text
exports\READY_TO_COPY
```

เมื่อรันเสร็จ ให้คัดลอกไฟล์ผลลัพธ์ทั้งหมดในโฟลเดอร์นี้ โดยเฉพาะไฟล์รวม:

```text
Server_Batches_004_008_All_Results.zip
```

โปรดอ่าน [README_TH.md](README_TH.md) สำหรับรายละเอียดการติดตั้ง การตรวจสถานะ การ pause/resume และวิธีนำผลกลับไปรวมอย่างครบถ้วน

## ขอบเขตแพ็กเกจ

- Batch 004–008 รวม 225 อาคาร
- ข้อมูล ground motion และฐานข้อมูลเริ่มต้นรวมอยู่ใน repository
- เก็บ SPO raw data, IDA checkpoints, IDA curves, fragility results, logs และ summary exports
- ไม่ต้องดาวน์โหลด Python packages ระหว่างติดตั้ง หากใช้ wheelhouse ที่ให้มา

อย่าคัดลอกเฉพาะฐานข้อมูล SQLite กลับเครื่องหลัก เพราะผลบางส่วนและหลักฐาน checkpoint อยู่ในโฟลเดอร์ผลลัพธ์อื่นด้วย
