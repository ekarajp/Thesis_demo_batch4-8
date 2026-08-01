# โปรแกรม Server สำหรับ Batch 004–008

แพ็กเกจนี้เป็นสำเนาแยกสำหรับสร้างข้อมูล Phase 1 เท่านั้น ได้แก่ Modal/SPO,
การเลือก CMS ตามคาบ, Full bidirectional IDA, PWSA Mandalay ที่ SF=1 และ
Fragility IO/LS/CP ของอาคารอันดับ 151–375 รวม 225 อาคาร ไม่มีการแบ่งชุด
Development/Validation/Test และไม่มีการฝึก ML ในแพ็กเกจนี้

## ขอบเขต Batch

- Batch 004: queue rank 151–200 (50 อาคาร)
- Batch 005: queue rank 201–250 (50 อาคาร)
- Batch 006: queue rank 251–300 (50 อาคาร)
- Batch 007: queue rank 301–350 (50 อาคาร)
- Batch 008: queue rank 351–375 (25 อาคาร)

## วิธีติดตั้งและเริ่มรัน

1. แตก ZIP ทั้งโฟลเดอร์ลงไดรฟ์ข้อมูล เช่น `D:` หรือ `E:` ห้ามรันจาก Drive C
2. ติดตั้ง 64-bit CPython 3.10 และ Windows Python Launcher (`py`) ถ้ายังไม่มี
3. ดับเบิลคลิก `START_UI.bat`
4. ครั้งแรกโปรแกรมจะสร้าง `.venv`, ติดตั้ง dependency ที่ล็อกไว้, ตรวจ
   checksum และเปิด `http://127.0.0.1:8765`
5. เลือกจำนวน workers (ค่าเริ่มต้น 6) แล้วกด **เริ่ม / ทำต่อ**

หน้าเว็บแสดงสถานะรวม สถานะแต่ละ Batch และเปอร์เซ็นต์รายอาคารจากข้อมูลที่
บันทึกจริง การปิดหน้าเว็บไม่ทำให้ตัว Batch runner หยุด แต่ไม่ควรปิดหน้าต่าง
Command Prompt ที่เป็น dashboard ระหว่างที่ต้องการใช้ UI

## การหยุดและทำต่อ

กด **หยุดชั่วคราวอย่างปลอดภัย** ในหน้าเว็บ โปรแกรมจะหยุด stage และ process
workers ทั้งหมด เก็บ checkpoint ที่เสร็จแล้ว และสร้าง SQLite backup
เมื่อเปิด `START_UI.bat` ใหม่ ให้กด **เริ่ม / ทำต่อ** โปรแกรมจะตรวจผลเดิมและ
ทำเฉพาะสิ่งที่ยังไม่เสร็จ ไม่เริ่ม Full IDA ใหม่ทั้งหมด

หากเครื่องดับโดยไม่ทันกด Pause ให้เปิด UI แล้วกด **เริ่ม / ทำต่อ** เช่นกัน
ฐานข้อมูลใช้ WAL และผล NLTHA แยกเป็นไฟล์ checkpoint

## นโยบายบันทึกทันที

- SPO: เมื่ออาคารหนึ่งวิเคราะห์เสร็จ จะเขียน raw curve/mechanism และ commit
  แถวของอาคารนั้นทันที ไม่รอครบทั้ง Batch
- Full IDA: ทุก target IM/NLTHA ที่เสร็จจะเขียน JSON แบบ atomic แล้ว `fsync`
  ลงดิสก์ทันที แม้ IDA curve นั้นยังไม่ครบ IO/LS/CP
- เมื่อเปิดทำต่อ Controller จะตรวจ analysis signature และอ่าน JSON ของจุด
  ที่เสร็จแล้วกลับมาใช้ จึงคำนวณเฉพาะ target IM ที่ยังขาด
- เมื่อ IDA curve หนึ่งคู่ GM เสร็จ จะ commit runs, capacities และ diagnostics
  ของคู่นั้นด้วย transaction เดียวทันที
- Fragility: fit และ commit ทีละอาคาร ไม่รอครบทั้ง Batch
- SQLite ของแพ็กเกจ Server ใช้ `synchronous=FULL` และสร้าง online backup
  ทุก 10 นาที รวมถึงหลังจบทุก stage และเมื่อกด Pause

ใน UI ช่อง `NLTHA checkpoints` นับไฟล์ที่บันทึกจริงบนดิสก์ด้วย จึงมองเห็น
งานที่เสร็จแล้วแม้ IDA curve ปัจจุบันยังไม่ปิดครบสาม limit states

## เมื่อ SPO อาคารหนึ่งหาคำตอบที่ valid ไม่ได้

โปรแกรมไม่ถือ numerical timeout/nonconvergence เป็นการพังของอาคาร และไม่ส่ง
อาคารนั้นเข้า IDA โดยฝืนเกณฑ์ แต่จะดำเนินการดังนี้โดยอัตโนมัติ:

1. เก็บแถว failure, partial SPO curve และ mechanism history เดิมไว้ใน
   `spo_quarantine` เพื่อรอตรวจซ่อมภายหลัง
2. คง queue slot เดิม แต่เลือกโมเดล valid ที่ยังไม่เคยใช้และใกล้ที่สุดจาก
   catalog อาคาร 5 ชั้นทั้งหมด 6,908 แบบ โดยเลือกได้จาก feasible unused
   models 6,526 แบบ ใช้เฉพาะ geometry, material, load,
   strength tiers, SCWB class/ratio และ axial ratio ไม่ใช้ผล SPO/IDA/fragility
3. ทำ SPO ของตัวแทน และทำซ้ำด้วย candidate ถัดไปหากยังไม่ valid จนได้จำนวน
   active SPO ครบตาม Batch
4. อาคารที่ SPO valid แล้วจะไม่ถูกรันซ้ำ เมื่ออาคารใดเลือก GM ผ่านเกณฑ์
   โปรแกรมจะส่งอาคารนั้นเข้า Full IDA ทันที โดยไม่รอ SPO ของทั้ง Batch
   ขณะเดียวกัน SPO อาคารถัดไปยังทำต่อด้วย process แยก

## Streaming SPO → Full IDA → Fragility

- ใช้ SPO producer 1 process และ Full-IDA consumer pool ค่าเริ่มต้น 6 workers
  พร้อมกัน จึงใช้ทรัพยากรเครื่องได้ต่อเนื่อง
- GM selection ทำเฉพาะ Building IDs ที่มี SPO valid แล้วเท่านั้น
- ถ้า IDA wave กำลังทำงาน อาคารที่ SPO เสร็จใหม่จะเข้าคิวและเริ่มใน IDA wave
  ถัดไปทันทีที่ worker pool ว่าง ไม่ต้องรอให้ SPO ครบทั้ง Batch
- เมื่อ Full IDA ของอาคารหนึ่งครบ primary CMS และ PWSA sensitivity แล้ว
  โปรแกรมจะ fit Fragility IO/LS/CP ของอาคารนั้นทันที แม้ IDA ของอาคารอื่น
  ยังรันอยู่ ไม่รอครบทั้ง Batch
- การ Pause/ไฟดับ/เปิดทำต่อ อาศัยผลที่บันทึกจริงใน SQLite และ raw checkpoint
  เท่านั้น ไม่อาศัยเพียงสถานะในหน้า UI จึงไม่รันอาคารหรือจุด IM ที่เสร็จแล้วซ้ำ

UI จะแสดงจำนวน `SPO quarantine`, generation ของตัวแทน และ Building ID
ต้นฉบับ ประวัติทั้งหมดถูกเขียนลง `spo_replacement_history` แบบ deterministic
เพื่อให้ตรวจสอบย้อนกลับได้ และ CSV/ไฟล์ raw ของ quarantine จะรวมอยู่ใน ZIP
ผลลัพธ์ด้วย

## ตำแหน่งผลลัพธ์

- ตารางและหน้าอ่านง่าย: `outputs\batch_004` ถึง `outputs\batch_008`
- Log แยก stage: `logs\batch_004` ถึง `logs\batch_008`
- ZIP ที่พร้อมก๊อปกลับ: `exports\READY_TO_COPY`
- สำรองฐานข้อมูลล่าสุด: `backups\server_latest.sqlite`

เมื่อแต่ละ Batch เสร็จ จะได้ไฟล์ เช่น:

`Batch_004_Result_Ranks_151_200.zip`

เมื่อครบทั้งหมด จะมี:

`Server_Batches_004_008_All_Results.zip`

## เมื่อรันเสร็จ ต้องไปก๊อปไฟล์จากที่ไหน

ให้เปิดโฟลเดอร์โปรแกรมที่แตก ZIP ไว้ แล้วเข้าไปที่:

```text
exports\READY_TO_COPY
```

ตัวอย่าง หากแตกโปรแกรมไว้ที่:

```text
D:\RC_Fragility_Server_Batches_04_08
```

Output ที่ต้องก๊อปจะอยู่ที่:

```text
D:\RC_Fragility_Server_Batches_04_08\exports\READY_TO_COPY
```

หากต้องการนำผลกลับมาทีละ Batch ให้ก๊อปไฟล์ต่อไปนี้:

```text
Batch_004_Result_Ranks_151_200.zip
Batch_005_Result_Ranks_201_250.zip
Batch_006_Result_Ranks_251_300.zip
Batch_007_Result_Ranks_301_350.zip
Batch_008_Result_Ranks_351_375.zip
```

หาก Batch 004–008 เสร็จครบทั้งหมดแล้ว ให้ก๊อปเพียงไฟล์เดียว:

```text
Server_Batches_004_008_All_Results.zip
```

ไฟล์รวมนี้มี ZIP ของทั้งห้า Batch อยู่ภายใน และสามารถส่งเข้า
`import_results.py` ได้โดยตรง ไม่ต้องแตก ZIP ผลลัพธ์ก่อน

อย่าก๊อปเพียงฐานข้อมูล `data\server_batches_004_008.sqlite` เพราะไฟล์นั้น
ไม่มี raw SPO/NLTHA files ครบในตัวเอง ให้ใช้ไฟล์จาก
`exports\READY_TO_COPY` เท่านั้น

ในหน้า UI หัวข้อ **ไฟล์พร้อมก๊อปกลับ** จะมีลิงก์ดาวน์โหลด ZIP ที่เสร็จแล้ว
เช่นกัน หากยังไม่มีไฟล์ในหัวข้อนี้ แสดงว่า Batch นั้นยังไม่ผ่านขั้นตอน
Fragility และ Export ครบถ้วน

ZIP ผลลัพธ์มี SQLite เฉพาะ Batch, CSV ทุกตาราง, raw SPO curve,
mechanism history, IDA capacities/diagnostics และ NLTHA checkpoint files
พร้อม checksum และ relocation manifest

## วิธีรวมผลกลับเครื่องหลัก

วาง ZIP ผลลัพธ์ไว้ในไดรฟ์ข้อมูล แล้วเปิด PowerShell จากโฟลเดอร์แพ็กเกจนี้:

```powershell
.\.venv\Scripts\python.exe .\import_results.py `
  --main-project "D:\QCE-NAS\Ekaraj Private Data\Civil Engineering\phD\Research\Demo Program" `
  --result-zip "E:\ServerResults\Server_Batches_004_008_All_Results.zip"
```

ตัวนำเข้าจะทำตามลำดับนี้:

1. ตรวจ ZIP checksum และป้องกัน path traversal
2. ตรวจ scientific config, source code, Ground Motion, controller model และ
   `Building ID / queue rank / model hash`
3. ตรวจว่าผลเดิมที่อาจมีอยู่ไม่ขัดแย้งกัน
4. ทำ online backup ของฐานข้อมูลเครื่องหลัก
5. ก๊อป raw files และตรวจ checksum อีกครั้ง
6. รวม SQLite ด้วย transaction เดียว หากเกิดข้อผิดพลาดจะ rollback

หาก config หรือ source code ของเครื่องหลักเปลี่ยนไป ตัวนำเข้าจะหยุดและไม่
ฝืนรวมผล เพื่อไม่ให้ข้อมูลจากคนละแบบจำลองปะปนกัน

## ข้อกำหนดสำคัญ

- รองรับ Windows 64-bit เพราะใช้ `openseespywin==3.5.1.3`
- ต้องใช้ CPython 3.10.x 64-bit เท่านั้น
- ห้ามแก้ `config\poc.json`, `src`, Ground Motion หรือ frozen controller
  ระหว่างการรัน
- ห้ามรันแพ็กเกจเดียวกันสอง instance พร้อมกัน
- UI เปิดเฉพาะ `127.0.0.1` จึงไม่ถูกเผยแพร่บนเครือข่ายโดยปริยาย
