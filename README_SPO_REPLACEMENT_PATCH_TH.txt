SPO quarantine + nearest replacement patch
==========================================

ใช้กับแพ็กเกจ RC_Fragility_Server_Batches_04_08 ที่กำลังรันอยู่ โดยไม่ทับ
ฐานข้อมูล ผล SPO/IDA หรือ checkpoint เดิม

วิธีติดตั้งบน Server

1. ใน Dashboard กด "หยุดชั่วคราวอย่างปลอดภัย"
2. รอจนสถานะเป็น paused และไม่มี Python worker ทำงาน
3. สำรองทั้งโฟลเดอร์ Server ไว้หนึ่งชุด
4. แตกไฟล์ patch แล้วก๊อปไฟล์/โฟลเดอร์ทั้งหมดมาทับที่ root ของโปรแกรมเดิม
   ห้ามลบ data\server_batches_004_008.sqlite, runs, outputs หรือ runtime เดิม
5. ดับเบิลคลิก APPLY_SPO_REPLACEMENT_PATCH.bat
6. ต้องเห็น PATCH PASSED
7. เปิด START_UI.bat แล้วกด "เริ่ม / ทำต่อ"

ผลที่คาดสำหรับปัญหาปัจจุบัน

- B-27a2f5ef33a0a1fe จะถูกเก็บใน spo_quarantine ไม่ถูกลบหรือถือเป็น collapse
- queue rank 182 จะใช้ candidate ใกล้ที่สุดที่ยังไม่เคยใช้
- candidate มาจาก catalog อาคาร 5 ชั้นทั้งหมด โดย runtime เลือกเฉพาะ
  feasible model ที่ valid และยังไม่ถูกใช้ในชุด 375 อาคาร
- 49 SPO ที่ valid อยู่แล้วจะถูก reuse ไม่รันใหม่
- หาก candidate ใหม่ยังทำ SPO ไม่ผ่าน โปรแกรมจะเลือก candidate ถัดไป
- 49 อาคารที่ SPO valid และเลือก GM ผ่านจะเริ่ม Full IDA ได้ทันที
  โดยไม่รอ SPO ของ candidate ใหม่; SPO และ IDA ทำงานซ้อนกัน
- อาคารใด Full IDA เสร็จจะทำ Fragility IO/LS/CP ต่อทันที
  โดยไม่รอ IDA ของอาคารอื่นครบ

ไฟล์ตรวจสอบ:

- runtime\spo_replacement_patch_install.json
- runtime\events.jsonl
- backups\server_latest.sqlite
- outputs\batch_004\spo_quarantine.csv (เมื่อ export)
- outputs\batch_004\spo_replacement_history.csv (เมื่อ export)
