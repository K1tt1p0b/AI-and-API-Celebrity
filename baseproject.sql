-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: localhost    Database: db_miniprojectfinal
-- ------------------------------------------------------
-- Server version	9.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `brand`
--

DROP TABLE IF EXISTS `brand`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `brand` (
  `brandID` int NOT NULL AUTO_INCREMENT,
  `brandName` varchar(255) NOT NULL,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`brandID`),
  UNIQUE KEY `brandName` (`brandName`)
) ENGINE=InnoDB AUTO_INCREMENT=193 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `brand`
--

LOCK TABLES `brand` WRITE;
/*!40000 ALTER TABLE `brand` DISABLE KEYS */;
INSERT INTO `brand` VALUES (1,'Dior','2025-07-31 09:42:02','2025-07-31 09:42:02'),(2,'MAC Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(3,'Maybelline','2025-07-31 09:42:02','2025-07-31 09:42:02'),(4,'L\'Oréal Paris','2025-07-31 09:42:02','2025-07-31 09:42:02'),(5,'Fenty Beauty','2025-07-31 09:42:02','2025-07-31 09:42:02'),(6,'Sephora Collection','2025-07-31 09:42:02','2025-07-31 09:42:02'),(7,'NARS Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(8,'Kylie Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(9,'Anastasia Beverly Hills','2025-07-31 09:42:02','2025-07-31 09:42:02'),(10,'Benefit Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(11,'MAC','2025-09-24 14:26:55','2025-09-24 14:26:55'),(16,'NARS','2025-09-24 14:26:55','2025-09-24 14:26:55'),(50,'Urban Decay','2025-09-25 12:21:21','2025-09-25 12:21:21'),(55,'Too Faced','2025-09-25 12:21:21','2025-09-25 12:21:21');
/*!40000 ALTER TABLE `brand` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `cosmetics`
--

DROP TABLE IF EXISTS `cosmetics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cosmetics` (
  `CosmeticID` int NOT NULL AUTO_INCREMENT,
  `Name` varchar(255) NOT NULL,
  `ShadeCode` varchar(50) DEFAULT NULL,
  `ShadeName` varchar(255) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Description` text,
  `Price` decimal(10,2) NOT NULL,
  `ImageURL` varchar(512) DEFAULT NULL,
  `ProductLink` varchar(512) DEFAULT NULL,
  `BrandID` int NOT NULL,
  `suitableSkinTone` varchar(50) DEFAULT NULL,
  `suitableBudgetRange` varchar(50) DEFAULT NULL,
  `suitableLookType` varchar(255) DEFAULT NULL,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`CosmeticID`),
  KEY `BrandID` (`BrandID`),
  CONSTRAINT `cosmetics_ibfk_1` FOREIGN KEY (`BrandID`) REFERENCES `brand` (`brandID`),
  CONSTRAINT `fk_cosmetics_brand` FOREIGN KEY (`BrandID`) REFERENCES `brand` (`brandID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cosmetics`
--

LOCK TABLES `cosmetics` WRITE;
/*!40000 ALTER TABLE `cosmetics` DISABLE KEYS */;
INSERT INTO `cosmetics` VALUES (2,'Studio Fix Fluid','NC15',NULL,'Foundation','Medium to full coverage foundation',560.00,NULL,'https://shopee.co.th/%F0%9F%92%8ECOD-%E0%B9%81%E0%B8%97%E0%B9%89%F0%9F%8E%80%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B9%83%E0%B8%99%E0%B8%81%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B9%80%E0%B8%97%E0%B8%9E%E0%B8%AF-MAC-Studio-Fix-Fluid-Foundation-SPF15-PA-N18-N12-NC15-NC20-30ml-i.1413382706.28070988400?sp_atk=2554d7c6-faf0-466c-bccf-cfa4d1dbbaf0&xptdk=2554d7c6-faf0-466c-bccf-cfa4d1dbbaf0',11,'Fair',NULL,'สายฝอ','2025-09-24 14:26:55','2025-09-25 14:24:17'),(3,'Studio Fix Fluid','NC30',NULL,'Foundation','Medium to full coverage foundation',894.00,NULL,'https://shopee.co.th/%E0%B9%81%E0%B8%97%E0%B9%89%F0%9F%92%AF-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99-MAC-studio-fix-fluid-spf-15-foundation-15ml-30ml-%E0%B8%AA%E0%B8%B5-Nc20-Nc25-Nc30-i.20177639.18938133874?sp_atk=87038f79-d447-4eee-afe5-39f71e7fccb2&xptdk=87038f79-d447-4eee-afe5-39f71e7fccb2',11,'Medium',NULL,'Full Glam ปกปิดแน่น (สายฝอ)','2025-09-24 14:26:55','2025-09-25 16:27:03'),(4,'Pro Longwear','NC20',NULL,'Concealer','Long-wearing concealer',1110.00,NULL,'https://shopee.co.th/M.A.C-Lipstick-%E0%B8%AA%E0%B8%B5%E0%B8%8A%E0%B8%A1%E0%B8%9E%E0%B8%B9%E0%B8%99%E0%B8%B9%E0%B9%89%E0%B8%94-%E0%B8%AA%E0%B8%B5%E0%B9%81%E0%B8%94%E0%B8%87-Rubywoo-612RussianRed-602-Chili-%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B9%81%E0%B8%A1%E0%B8%97Matte-3g-i.1203846781.40060675018?sp_atk=33967b2e-b4dd-4cf6-8e8c-1bd9294edfc2&xptdk=33967b2e-b4dd-4cf6-8e8c-1bd9294edfc2',11,'Fair',NULL,'งานผิวเนียน สุภาพ ทำงาน/เรียน','2025-09-24 14:26:55','2025-09-25 16:27:03'),(5,'Ruby Woo',NULL,'Red','Lipstick','Matte red lipstick',318.00,'/images/MAC_Lipstick_Ruby_Woo.jpeg','https://shopee.co.th/M%E2%80%A2A%E2%80%A2C-PRO-LONGWEAR-CONCEALER-9ML-.3FLOZ-i.54549229.40468529163?sp_atk=b6f6abe6-b748-4c16-ae5a-eee844228716&xptdk=b6f6abe6-b748-4c16-ae5a-eee844228716',11,'All',NULL,'Classic Glam ปากแดง (สายฝอ/ออกงาน)','2025-09-24 14:26:55','2025-09-25 16:27:03'),(6,'Melba',NULL,'Coral Pink','Blush','Matte coral pink blush',1099.00,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-MAC-Powder-Blush-Sheertone-Shimmer-Blush-Mineralize-Blush-i.1300084.3626655175?sp_atk=f6045098-2acf-4df0-9b12-c1ded7f4c2d2&xptdk=f6045098-2acf-4df0-9b12-c1ded7f4c2d2',11,'Medium',NULL,'สายเกาหลี (จะเข้าธรรมชาติก็ได้)','2025-09-24 14:26:55','2025-09-25 14:24:17'),(7,'Natural Radiant Longwear',NULL,'Gobi','Foundation','Natural radiant coverage',2618.00,NULL,'https://shopee.co.th/Nars-Natural-Radiant-Longwear-Foundation-30ml.-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B8%9A%E0%B8%B2%E0%B8%87%E0%B9%80%E0%B8%9A%E0%B8%B2-%E0%B8%AA%E0%B8%B3%E0%B8%AB%E0%B8%A3%E0%B8%B1%E0%B8%9A%E0%B8%9C%E0%B8%B4%E0%B8%A7%E0%B8%AB%E0%B8%99%E0%B9%89%E0%B8%B2-i.195689293.27177233484?sp_atk=cc8a248d-70d1-4e4c-a7ac-1d840612b5c1&xptdk=cc8a248d-70d1-4e4c-a7ac-1d840612b5c1',16,'Fair',NULL,'ธรรมชาติ (หรือ สายเกาหลี ก็ได้)','2025-09-24 14:26:55','2025-09-25 14:24:17'),(8,'Orgasm',NULL,'Peachy Pink','Blush','Iconic peachy pink with golden undertones',427.00,NULL,'https://shopee.co.th/Nars-Blush-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C%E0%B8%AA-%E0%B8%9A%E0%B8%A5%E0%B8%B1%E0%B8%8A%E0%B8%AD%E0%B8%AD%E0%B8%99-%E0%B9%81%E0%B8%9A%E0%B8%9A-Orgasm-Behave-OrgasmX-Taj-Mahal-Sample-1.2g-i.1256941170.43818685453?sp_atk=8060ceb6-2a6b-4333-aad3-37439fb43490&xptdk=8060ceb6-2a6b-4333-aad3-37439fb43490',16,'All',NULL,'สายเกาหลี สดใสแก้มละมุน','2025-09-25 12:21:21','2025-09-25 16:27:03'),(9,'Radiant Creamy Concealer',NULL,'Vanilla','Concealer','Creamy full coverage concealer',1299.00,NULL,'https://shopee.co.th/NARS-Radiant-Creamy-Concealer-6ml-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C%E0%B8%AA-%E0%B8%84%E0%B8%AD%E0%B8%99%E0%B8%8B%E0%B8%B5%E0%B8%A5%E0%B9%80%E0%B8%A5%E0%B8%AD%E0%B8%A3%E0%B9%8C%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B8%AA%E0%B8%B1%E0%B8%A1%E0%B8%9C%E0%B8%B1%E0%B8%AA%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%AB%E0%B8%A3%E0%B8%B9%E0%B8%AB%E0%B8%A3%E0%B8%B2-i.70998059.1566168616?sp_atk=7b2871e5-4814-4933-a900-0c8b1433abc0&xptdk=7b2871e5-4814-4933-a900-0c8b1433abc0',16,'Fair',NULL,'สายฝอ ปกปิดแน่น','2025-09-25 12:21:21','2025-09-25 16:27:03'),(10,'Audacious',NULL,'Annabella','Lipstick','Full coverage lipstick',545.00,NULL,'https://shopee.co.th/-%E0%B8%AA%E0%B8%B8%E0%B8%94%E0%B8%84%E0%B8%B8%E0%B9%89%E0%B8%A1-nars-audacious-lipstick-i.95495597.8736501555?sp_atk=de085b0d-8555-4d48-9eed-84d328c5cc1f&xptdk=de085b0d-8555-4d48-9eed-84d328c5cc1f',16,'All',NULL,'Everyday Natural Lip','2025-09-25 12:21:21','2025-09-25 16:27:03'),(11,'Stay Naked','20NN',NULL,'Foundation','Weightless medium coverage',1190.00,NULL,'https://shopee.co.th/Urban-Decay-Stay-Naked-Foundation-%E0%B8%AA%E0%B8%B5-20NN-i.7984013.5617286830',50,'Fair',NULL,'งานผิวธรรมชาติ เรียบเนียน (Everyday Natural)','2025-09-25 12:21:21','2025-09-25 16:27:03'),(12,'Stay Naked','40NN',NULL,'Foundation','Weightless medium coverage',1490.00,NULL,'https://shopee.co.th/%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B8%9F%E0%B8%A3%E0%B8%B5-Urban-Decay-Stay-Naked-Foundation-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99-31NN-40NN-60WY-61NN-i.540449302.9794737212',50,'Medium',NULL,'งานผิวธรรมชาติ เรียบเนียน (Everyday Natural)','2025-09-25 12:21:21','2025-09-25 16:27:03'),(13,'Vice Lipstick','714',NULL,'Lipstick','Creamy lipstick',200.00,NULL,'https://shopee.co.th/Urban-Decay-Vice-Lipstick-714-i.17996148.771441688',50,'All',NULL,'Everyday Natural Lip','2025-09-25 12:21:21','2025-09-25 16:27:03'),(14,'Stay Naked Concealer','20NN',NULL,'Concealer','Full coverage concealer',550.00,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-*%E0%B8%A5%E0%B8%94%E0%B8%A5%E0%B9%89%E0%B8%B2%E0%B8%87%E0%B8%AA%E0%B8%95%E0%B9%87%E0%B8%AD%E0%B8%84*-URBAN-DECAY-Stay-Naked-Concealer-i.1300084.2815835505',50,'Fair',NULL,'สายฝอ ปกปิดแน่น','2025-09-25 12:21:21','2025-09-25 16:27:03'),(15,'Stay Naked Powder',NULL,'Fair Neutral','Powder','Weightless finishing powder',1300.00,NULL,'https://shopee.co.th/URBAN-DECAY-%E0%B9%81%E0%B8%9B%E0%B9%89%E0%B8%87-Stay-Naked-Pressed-Powder-i.14275840.15649559636',50,'Fair',NULL,'งานผิวธรรมชาติ เซ็ตเมคอัพ','2025-09-25 12:21:21','2025-09-25 16:27:03'),(16,'Born This Way',NULL,'Vanilla','Foundation','Full coverage foundation',2190.00,NULL,'https://shopee.co.th/Too-Faced-Born-This-Way-Matte-24-Hour-Long-Wear-Foundation-30ml-i.35060332.19472911329?sp_atk=993a22dd-3bd2-47d3-a59c-1493dd006834&xptdk=993a22dd-3bd2-47d3-a59c-1493dd006834',55,'Fair',NULL,'Full Glam ปกปิดแน่น (สายฝอ)','2025-09-25 12:21:21','2025-09-25 16:27:03'),(17,'Born This Way',NULL,'Light Beige','Foundation','Full coverage foundation',2190.00,NULL,'https://shopee.co.th/Too-Faced-Born-This-Way-Matte-24-Hour-Long-Wear-Foundation-30ml-i.35060332.19472911329?sp_atk=5a9a8a3f-f588-4a7b-87d1-6123570b9fac&xptdk=5a9a8a3f-f588-4a7b-87d1-6123570b9fac',55,'Medium',NULL,'Full Glam ปกปิดแน่น (สายฝอ)','2025-09-25 12:21:21','2025-09-25 16:27:03'),(18,'Born This Way Super Coverage',NULL,'Light','Concealer','Multi-use concealer',1741.00,NULL,'https://shopee.co.th/%F0%9F%87%BA%F0%9F%87%B8Preorder%F0%9F%87%BA%F0%9F%87%B8-Too-Faced-Born-This-Way-Super-Coverage-Multi-Use-Concealer-%E0%B9%81%E0%B8%97%E0%B9%89100--i.59311125.27660729813',55,'Fair',NULL,'งานผิวเนียน สุภาพ ทำงาน/เรียน','2025-09-25 12:21:21','2025-09-25 16:27:03'),(19,'Melted Matte',NULL,'Peach','Blush','Long-wearing blush',452.00,NULL,'https://shopee.co.th/%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B5%E0%B8%9C%E0%B8%A1%E0%B8%95%E0%B8%A3%E0%B8%87-Too-Faced-Rabbit-%E0%B8%99%E0%B8%B9%E0%B8%99-Blush-%E0%B9%80%E0%B8%AA%E0%B8%A3%E0%B8%B4%E0%B8%A1-Complexion-Natural-Matte-Drunk-Stage-82AO-i.448456413.41612064938?sp_atk=d2858db3-4cdd-49c3-8bae-4c8e4fce0555&xptdk=d2858db3-4cdd-49c3-8bae-4c8e4fce0555',55,'Medium',NULL,'สายเกาหลี สดใสแก้มละมุน','2025-09-25 12:21:21','2025-09-25 16:27:03'),(20,'Born This Way Powder',NULL,'Light','Powder','Setting powder',1790.00,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-%E0%B9%81%E0%B8%97%E0%B9%89-%F0%9F%92%AF-Toofaced-Born-This-Way-The-Natural-Nudes-Palette-Born-Like-This-Palette-i.184143361.11015675616?sp_atk=2c1ec770-40d1-4563-bb1c-dedf2c00cb8d&xptdk=2c1ec770-40d1-4563-bb1c-dedf2c00cb8d',55,'Fair',NULL,'งานผิวธรรมชาติ เซ็ตเมคอัพ','2025-09-25 12:21:21','2025-09-25 16:27:03');
/*!40000 ALTER TABLE `cosmetics` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `feedback`
--

DROP TABLE IF EXISTS `feedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `feedback` (
  `FeedbackID` int NOT NULL AUTO_INCREMENT,
  `CommentText` text,
  `Rating` int NOT NULL,
  `Date` date DEFAULT (curdate()),
  `Users_ID` int NOT NULL,
  `CosmeticID` int DEFAULT NULL,
  PRIMARY KEY (`FeedbackID`),
  KEY `fk_feedback_user` (`Users_ID`),
  KEY `idx_feedback_user` (`Users_ID`),
  KEY `idx_feedback_cosmetic` (`CosmeticID`),
  CONSTRAINT `fk_feedback_user` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `chk_feedback_rating` CHECK ((`Rating` between 1 and 5))
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `feedback`
--

LOCK TABLES `feedback` WRITE;
/*!40000 ALTER TABLE `feedback` DISABLE KEYS */;
INSERT INTO `feedback` VALUES (1,'แอปดีมากเลยค่ะ ชอบสุดๆ!',5,'2025-08-14',17,NULL),(2,'TEst',3,'2025-08-14',17,NULL),(3,'asd',4,'2025-09-22',18,NULL),(4,'good',5,'2025-09-24',18,NULL);
/*!40000 ALTER TABLE `feedback` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `makeuplook`
--

DROP TABLE IF EXISTS `makeuplook`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `makeuplook` (
  `LookID` int NOT NULL AUTO_INCREMENT,
  `lookName` varchar(255) NOT NULL,
  `lookCategory` varchar(255) DEFAULT NULL,
  `description` text,
  PRIMARY KEY (`LookID`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `makeuplook`
--

LOCK TABLES `makeuplook` WRITE;
/*!40000 ALTER TABLE `makeuplook` DISABLE KEYS */;
INSERT INTO `makeuplook` VALUES (1,'ธรรมชาติ','Everyday Looks','ธรรมชาติใสๆ'),(2,'สายเกาหลี','Cultural Styles','เน้นผิวฉ่ำวาว อายไลเนอร์บางเบา สีปากสดใส'),(3,'สายฝอ','Cultural Styles','เน้นโครงหน้าชัด คอนทัวร์หนัก อายแชโดว์คมเข้ม'),(4,'สโมคกี้อายส์','Party Looks','เน้นดวงตาโดดเด่น ด้วยอายแชโดว์สีเข้ม'),(5,'Everyday Glam','Everyday Looks','ลุคที่ดูแต่งหน้า แต่ยังคงความเบาและสดใส เหมาะกับทุกวัน'),(6,'งานกลางคืน','Party Looks','ลุคสำหรับออกงานกลางคืน หรูหราและโดดเด่น'),(7,'วินเทจ','Thematic Looks','สไตล์การแต่งหน้าย้อนยุค เช่น ยุค 60s, 70s'),(8,'เทรนดี้','Seasonal Looks','ลุคที่กำลังอินเทรนด์ในปัจจุบัน'),(9,'งานรับปริญญา','Special Occasion','ลุคที่สุภาพ แต่ยังคงความสวยงาม เหมาะกับวันสำคัญ'),(10,'แนวพังก์','Alternative Styles','ลุคที่เน้นความขบถและมีเอกลักษณ์เฉพาะตัว');
/*!40000 ALTER TABLE `makeuplook` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `recommendation`
--

DROP TABLE IF EXISTS `recommendation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `recommendation` (
  `RecommendationID` int NOT NULL AUTO_INCREMENT,
  `Users_ID` int NOT NULL,
  `CosmeticID` int NOT NULL,
  `MatchPercentage` decimal(5,2) DEFAULT NULL,
  `ContextJSON` json DEFAULT NULL,
  `CreatedAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`RecommendationID`),
  KEY `idx_rec_user` (`Users_ID`),
  KEY `idx_rec_cosmetic` (`CosmeticID`),
  CONSTRAINT `fk_rec_cosmetic` FOREIGN KEY (`CosmeticID`) REFERENCES `cosmetics` (`CosmeticID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_rec_user` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `recommendation`
--

LOCK TABLES `recommendation` WRITE;
/*!40000 ALTER TABLE `recommendation` DISABLE KEYS */;
/*!40000 ALTER TABLE `recommendation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `recommendedcolorpalettes`
--

DROP TABLE IF EXISTS `recommendedcolorpalettes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `recommendedcolorpalettes` (
  `PaletteID` int NOT NULL AUTO_INCREMENT,
  `PaletteName` varchar(255) NOT NULL,
  `SuitableSkinTone` varchar(50) NOT NULL,
  `ImageURL` varchar(512) NOT NULL,
  `Description` text,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`PaletteID`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `recommendedcolorpalettes`
--

LOCK TABLES `recommendedcolorpalettes` WRITE;
/*!40000 ALTER TABLE `recommendedcolorpalettes` DISABLE KEYS */;
INSERT INTO `recommendedcolorpalettes` VALUES (1,'Warm Tone General Color Palette','Warm Tone','medium.jpg','ตารางสีสำหรับผิวโทนอุ่น','2025-07-31 14:17:12','2025-09-25 14:51:25'),(2,'Cool Tone General Color Palette','Cool Tone','fair.jpg','ตารางสีสำหรับผิวโทนเย็น','2025-07-31 14:17:12','2025-09-25 14:48:00'),(3,'Neutral Tone General Color Palette','Neutral Tone','dark.jpg','ตารางสีสำหรับผิวโทนกลาง','2025-07-31 14:17:12','2025-09-25 14:55:43');
/*!40000 ALTER TABLE `recommendedcolorpalettes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `retailer_offers`
--

DROP TABLE IF EXISTS `retailer_offers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `retailer_offers` (
  `OfferID` bigint NOT NULL AUTO_INCREMENT,
  `CosmeticID` int NOT NULL,
  `Retailer` enum('shopee','lazada','sephora','watsons','other','legacy') NOT NULL,
  `RetailerShopID` varchar(64) DEFAULT NULL,
  `RetailerProductID` varchar(64) DEFAULT NULL,
  `URL` varchar(1024) NOT NULL,
  `ImageURL` varchar(1024) DEFAULT NULL,
  `PriceTHB` decimal(10,2) DEFAULT NULL,
  `Rating` decimal(3,2) DEFAULT NULL,
  `ReviewCount` int DEFAULT NULL,
  `IsOfficial` tinyint(1) NOT NULL DEFAULT '0',
  `LastUpdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`OfferID`),
  UNIQUE KEY `uq_offer` (`CosmeticID`,`URL`(191)),
  UNIQUE KEY `uk_retailer_item` (`Retailer`,`RetailerProductID`),
  KEY `idx_cosmetic` (`CosmeticID`),
  KEY `idx_offer_rank` (`CosmeticID`,`IsOfficial`,`PriceTHB`),
  CONSTRAINT `fk_offer_cosmetic` FOREIGN KEY (`CosmeticID`) REFERENCES `cosmetics` (`CosmeticID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `chk_offers_price` CHECK ((`PriceTHB` >= 0))
) ENGINE=InnoDB AUTO_INCREMENT=74 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `retailer_offers`
--

LOCK TABLES `retailer_offers` WRITE;
/*!40000 ALTER TABLE `retailer_offers` DISABLE KEYS */;
INSERT INTO `retailer_offers` VALUES (1,2,'shopee',NULL,NULL,'https://shopee.co.th/%F0%9F%92%8ECOD-%E0%B9%81%E0%B8%97%E0%B9%89%F0%9F%8E%80%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B9%83%E0%B8%99%E0%B8%81%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B9%80%E0%B8%97%E0%B8%9E%E0%B8%AF-MAC-Studio-Fix-Fluid-Foundation-SPF15-PA-N18-N12-NC15-NC20-30ml-i.1413382706.28070988400?sp_atk=2554d7c6-faf0-466c-bccf-cfa4d1dbbaf0&xptdk=2554d7c6-faf0-466c-bccf-cfa4d1dbbaf0',NULL,1450.00,NULL,NULL,0,'2025-09-24 15:05:53'),(2,3,'shopee',NULL,NULL,'https://shopee.co.th/%E0%B9%81%E0%B8%97%E0%B9%89%F0%9F%92%AF-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99-MAC-studio-fix-fluid-spf-15-foundation-15ml-30ml-%E0%B8%AA%E0%B8%B5-Nc20-Nc25-Nc30-i.20177639.18938133874?sp_atk=87038f79-d447-4eee-afe5-39f71e7fccb2&xptdk=87038f79-d447-4eee-afe5-39f71e7fccb2',NULL,1450.00,NULL,NULL,0,'2025-09-24 15:05:53'),(3,5,'shopee',NULL,NULL,'https://shopee.co.th/M.A.C-Lipstick-%E0%B8%AA%E0%B8%B5%E0%B8%8A%E0%B8%A1%E0%B8%9E%E0%B8%B9%E0%B8%99%E0%B8%B9%E0%B9%89%E0%B8%94-%E0%B8%AA%E0%B8%B5%E0%B9%81%E0%B8%94%E0%B8%87-Rubywoo-612RussianRed-602-Chili-%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B9%81%E0%B8%A1%E0%B8%97Matte-3g-i.1203846781.40060675018?sp_atk=33967b2e-b4dd-4cf6-8e8c-1bd9294edfc2&xptdk=33967b2e-b4dd-4cf6-8e8c-1bd9294edfc2',NULL,890.00,NULL,NULL,0,'2025-09-24 15:05:53'),(4,4,'shopee',NULL,NULL,'https://shopee.co.th/M%E2%80%A2A%E2%80%A2C-PRO-LONGWEAR-CONCEALER-9ML-.3FLOZ-i.54549229.40468529163?sp_atk=b6f6abe6-b748-4c16-ae5a-eee844228716&xptdk=b6f6abe6-b748-4c16-ae5a-eee844228716',NULL,890.00,NULL,NULL,0,'2025-09-24 15:05:53'),(5,6,'shopee',NULL,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-MAC-Powder-Blush-Sheertone-Shimmer-Blush-Mineralize-Blush-i.1300084.3626655175?sp_atk=f6045098-2acf-4df0-9b12-c1ded7f4c2d2&xptdk=f6045098-2acf-4df0-9b12-c1ded7f4c2d2',NULL,890.00,NULL,NULL,0,'2025-09-24 15:05:53'),(6,7,'shopee',NULL,NULL,'https://shopee.co.th/Nars-Natural-Radiant-Longwear-Foundation-30ml.-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B8%9A%E0%B8%B2%E0%B8%87%E0%B9%80%E0%B8%9A%E0%B8%B2-%E0%B8%AA%E0%B8%B3%E0%B8%AB%E0%B8%A3%E0%B8%B1%E0%B8%9A%E0%B8%9C%E0%B8%B4%E0%B8%A7%E0%B8%AB%E0%B8%99%E0%B9%89%E0%B8%B2-i.195689293.27177233484?sp_atk=cc8a248d-70d1-4e4c-a7ac-1d840612b5c1&xptdk=cc8a248d-70d1-4e4c-a7ac-1d840612b5c1',NULL,1850.00,NULL,NULL,0,'2025-09-24 15:05:53'),(61,8,'shopee',NULL,NULL,'https://shopee.co.th/Nars-Blush-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C%E0%B8%AA-%E0%B8%9A%E0%B8%A5%E0%B8%B1%E0%B8%8A%E0%B8%AD%E0%B8%AD%E0%B8%99-%E0%B9%81%E0%B8%9A%E0%B8%9A-Orgasm-Behave-OrgasmX-Taj-Mahal-Sample-1.2g-i.1256941170.43818685453?sp_atk=8060ceb6-2a6b-4333-aad3-37439fb43490&xptdk=8060ceb6-2a6b-4333-aad3-37439fb43490',NULL,427.00,NULL,NULL,1,'2025-09-25 14:29:57'),(62,9,'shopee',NULL,NULL,'https://shopee.co.th/NARS-Radiant-Creamy-Concealer-6ml-%E0%B8%99%E0%B8%B2%E0%B8%A3%E0%B9%8C%E0%B8%AA-%E0%B8%84%E0%B8%AD%E0%B8%99%E0%B8%8B%E0%B8%B5%E0%B8%A5%E0%B9%80%E0%B8%A5%E0%B8%AD%E0%B8%A3%E0%B9%8C%E0%B9%80%E0%B8%99%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B8%AA%E0%B8%B1%E0%B8%A1%E0%B8%9C%E0%B8%B1%E0%B8%AA%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%AB%E0%B8%A3%E0%B8%B9%E0%B8%AB%E0%B8%A3%E0%B8%B2-i.70998059.1566168616?sp_atk=7b2871e5-4814-4933-a900-0c8b1433abc0&xptdk=7b2871e5-4814-4933-a900-0c8b1433abc0',NULL,1299.00,NULL,NULL,1,'2025-09-25 14:32:24'),(63,10,'shopee',NULL,NULL,'https://shopee.co.th/-%E0%B8%AA%E0%B8%B8%E0%B8%94%E0%B8%84%E0%B8%B8%E0%B9%89%E0%B8%A1-nars-audacious-lipstick-i.95495597.8736501555?sp_atk=de085b0d-8555-4d48-9eed-84d328c5cc1f&xptdk=de085b0d-8555-4d48-9eed-84d328c5cc1f',NULL,545.00,NULL,NULL,1,'2025-09-25 14:34:52'),(64,11,'shopee',NULL,NULL,'https://shopee.co.th/Urban-Decay-Stay-Naked-Foundation-%E0%B8%AA%E0%B8%B5-20NN-i.7984013.5617286830',NULL,1190.00,NULL,NULL,1,'2025-09-25 14:34:57'),(65,12,'shopee',NULL,NULL,'https://shopee.co.th/%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B8%9F%E0%B8%A3%E0%B8%B5-Urban-Decay-Stay-Naked-Foundation-%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99-31NN-40NN-60WY-61NN-i.540449302.9794737212',NULL,1490.00,NULL,NULL,1,'2025-09-25 14:35:00'),(66,13,'shopee',NULL,NULL,'https://shopee.co.th/Urban-Decay-Vice-Lipstick-714-i.17996148.771441688',NULL,200.00,NULL,NULL,1,'2025-09-25 14:35:23'),(67,14,'shopee',NULL,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-*%E0%B8%A5%E0%B8%94%E0%B8%A5%E0%B9%89%E0%B8%B2%E0%B8%87%E0%B8%AA%E0%B8%95%E0%B9%87%E0%B8%AD%E0%B8%84*-URBAN-DECAY-Stay-Naked-Concealer-i.1300084.2815835505',NULL,550.00,NULL,NULL,1,'2025-09-25 14:35:25'),(68,15,'shopee',NULL,NULL,'https://shopee.co.th/URBAN-DECAY-%E0%B9%81%E0%B8%9B%E0%B9%89%E0%B8%87-Stay-Naked-Pressed-Powder-i.14275840.15649559636',NULL,1300.00,NULL,NULL,1,'2025-09-25 14:35:28'),(69,16,'shopee',NULL,NULL,'https://shopee.co.th/Too-Faced-Born-This-Way-Matte-24-Hour-Long-Wear-Foundation-30ml-i.35060332.19472911329?sp_atk=993a22dd-3bd2-47d3-a59c-1493dd006834&xptdk=993a22dd-3bd2-47d3-a59c-1493dd006834',NULL,2190.00,NULL,NULL,1,'2025-09-25 14:35:30'),(70,17,'shopee',NULL,NULL,'https://shopee.co.th/Too-Faced-Born-This-Way-Matte-24-Hour-Long-Wear-Foundation-30ml-i.35060332.19472911329?sp_atk=5a9a8a3f-f588-4a7b-87d1-6123570b9fac&xptdk=5a9a8a3f-f588-4a7b-87d1-6123570b9fac',NULL,2190.00,NULL,NULL,1,'2025-09-25 14:35:34'),(71,18,'shopee',NULL,NULL,'https://shopee.co.th/%F0%9F%87%BA%F0%9F%87%B8Preorder%F0%9F%87%BA%F0%9F%87%B8-Too-Faced-Born-This-Way-Super-Coverage-Multi-Use-Concealer-%E0%B9%81%E0%B8%97%E0%B9%89100--i.59311125.27660729813',NULL,1741.00,NULL,NULL,1,'2025-09-25 14:35:36'),(72,19,'shopee',NULL,NULL,'https://shopee.co.th/%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B5%E0%B8%9C%E0%B8%A1%E0%B8%95%E0%B8%A3%E0%B8%87-Too-Faced-Rabbit-%E0%B8%99%E0%B8%B9%E0%B8%99-Blush-%E0%B9%80%E0%B8%AA%E0%B8%A3%E0%B8%B4%E0%B8%A1-Complexion-Natural-Matte-Drunk-Stage-82AO-i.448456413.41612064938?sp_atk=d2858db3-4cdd-49c3-8bae-4c8e4fce0555&xptdk=d2858db3-4cdd-49c3-8bae-4c8e4fce0555',NULL,452.00,NULL,NULL,1,'2025-09-25 14:35:40'),(73,20,'shopee',NULL,NULL,'https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-%E0%B9%81%E0%B8%97%E0%B9%89-%F0%9F%92%AF-Toofaced-Born-This-Way-The-Natural-Nudes-Palette-Born-Like-This-Palette-i.184143361.11015675616?sp_atk=2c1ec770-40d1-4563-bb1c-dedf2c00cb8d&xptdk=2c1ec770-40d1-4563-bb1c-dedf2c00cb8d',NULL,1790.00,NULL,NULL,1,'2025-09-25 14:35:44');
/*!40000 ALTER TABLE `retailer_offers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `role`
--

DROP TABLE IF EXISTS `role`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `role` (
  `Role_ID` int NOT NULL AUTO_INCREMENT,
  `Type_Name` varchar(50) NOT NULL,
  PRIMARY KEY (`Role_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `role`
--

LOCK TABLES `role` WRITE;
/*!40000 ALTER TABLE `role` DISABLE KEYS */;
INSERT INTO `role` VALUES (1,'user'),(2,'admin');
/*!40000 ALTER TABLE `role` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `similarity`
--

DROP TABLE IF EXISTS `similarity`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `similarity` (
  `similarity_ID` int NOT NULL AUTO_INCREMENT,
  `similarity_Date` date NOT NULL,
  `similarityDetail_Percent` decimal(5,2) DEFAULT NULL,
  `ThaiCelebrities_ID` int DEFAULT NULL,
  `Users_ID` int NOT NULL,
  PRIMARY KEY (`similarity_ID`),
  KEY `ThaiCelebrities_ID` (`ThaiCelebrities_ID`),
  KEY `fk_User_ID` (`Users_ID`),
  KEY `idx_similarity_celeb` (`ThaiCelebrities_ID`),
  CONSTRAINT `fk_similarity_celeb` FOREIGN KEY (`ThaiCelebrities_ID`) REFERENCES `thaicelebrities` (`ThaiCelebrities_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_similarity_user` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_User_ID` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`),
  CONSTRAINT `similarity_ibfk_1` FOREIGN KEY (`ThaiCelebrities_ID`) REFERENCES `thaicelebrities` (`ThaiCelebrities_ID`),
  CONSTRAINT `chk_similarity_percent` CHECK ((`similarityDetail_Percent` between 0 and 100))
) ENGINE=InnoDB AUTO_INCREMENT=102 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `similarity`
--

LOCK TABLES `similarity` WRITE;
/*!40000 ALTER TABLE `similarity` DISABLE KEYS */;
INSERT INTO `similarity` VALUES (67,'2025-02-25',35.26,81,17),(68,'2025-02-25',35.26,81,17),(69,'2025-02-25',29.50,36,17),(70,'2025-02-25',35.26,81,17),(71,'2025-02-25',33.71,67,17),(72,'2025-03-05',29.88,26,17),(73,'2025-03-05',30.59,26,17),(74,'2025-03-05',30.34,26,17),(75,'2025-03-05',30.36,26,17),(76,'2025-07-30',57.98,47,17),(77,'2025-07-31',46.59,59,17),(78,'2025-07-31',46.59,59,17),(79,'2025-07-31',46.14,59,17),(80,'2025-07-31',45.28,59,17),(81,'2025-07-31',44.86,59,17),(82,'2025-08-12',46.68,59,17),(83,'2025-08-12',46.68,59,17),(84,'2025-08-12',46.49,59,17),(85,'2025-08-12',45.02,59,17),(86,'2025-08-12',44.70,59,17),(87,'2025-08-12',46.73,59,17),(88,'2025-08-14',56.92,48,17),(89,'2025-08-15',55.70,91,18),(90,'2025-08-15',63.87,17,18),(91,'2025-08-15',68.60,17,18),(92,'2025-08-15',68.60,17,17),(93,'2025-08-15',68.60,17,17),(94,'2025-08-15',68.60,17,17),(95,'2025-08-15',68.60,17,17),(96,'2025-08-15',68.60,17,17),(97,'2025-08-15',64.82,10,17),(98,'2025-08-18',57.15,61,18),(99,'2025-08-18',63.87,17,18),(100,'2025-08-18',55.39,38,19),(101,'2025-08-18',60.81,63,19);
/*!40000 ALTER TABLE `similarity` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `skintoneanalysis`
--

DROP TABLE IF EXISTS `skintoneanalysis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `skintoneanalysis` (
  `SkinToneAnalysisID` int NOT NULL AUTO_INCREMENT,
  `SkinTone` varchar(100) DEFAULT NULL,
  `Users_ID` int DEFAULT NULL,
  `Undertone` varchar(20) DEFAULT NULL,
  `Confidence` tinyint DEFAULT NULL,
  PRIMARY KEY (`SkinToneAnalysisID`),
  KEY `Users_ID` (`Users_ID`),
  CONSTRAINT `fk_skin_user` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `skintoneanalysis_ibfk_1` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=95 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `skintoneanalysis`
--

LOCK TABLES `skintoneanalysis` WRITE;
/*!40000 ALTER TABLE `skintoneanalysis` DISABLE KEYS */;
INSERT INTO `skintoneanalysis` VALUES (1,'Cool Tone',17,NULL,NULL),(2,'Cool Tone',17,NULL,NULL),(3,'Neutral Tone',17,NULL,NULL),(4,'Neutral Tone',17,NULL,NULL),(5,'Neutral Tone',17,NULL,NULL),(6,'Neutral Tone',17,NULL,NULL),(7,'Neutral Tone',17,NULL,NULL),(8,'Neutral Tone',17,NULL,NULL),(9,'Warm Tone',17,NULL,NULL),(10,'Warm Tone',17,NULL,NULL),(11,'Neutral Tone',17,NULL,NULL),(12,'Warm Tone',17,NULL,NULL),(13,'Neutral Tone',17,NULL,NULL),(14,'Neutral Tone',17,NULL,NULL),(15,'Neutral Tone',17,NULL,NULL),(16,'Warm Tone',17,NULL,NULL),(17,'Warm Tone',17,NULL,NULL),(18,'Warm Tone',17,NULL,NULL),(19,'Warm Tone',17,NULL,NULL),(20,'Warm Tone',17,NULL,NULL),(21,'Neutral Tone',17,NULL,NULL),(22,'Neutral Tone',17,NULL,NULL),(23,'Warm Tone',17,NULL,NULL),(24,'Neutral Tone',18,NULL,NULL),(25,'Neutral Tone',18,NULL,NULL),(26,'Neutral Tone',18,NULL,NULL),(27,'Neutral Tone',18,NULL,NULL),(28,'Neutral Tone',18,NULL,NULL),(29,'Neutral Tone',18,NULL,NULL),(30,'Cool Tone',18,NULL,NULL),(31,'Cool Tone',18,NULL,NULL),(32,'Cool Tone',18,NULL,NULL),(33,'Neutral Tone',17,NULL,NULL),(34,'Neutral Tone',17,NULL,NULL),(35,'Neutral Tone',17,NULL,NULL),(36,'Neutral Tone',18,NULL,NULL),(37,'Neutral Tone',18,NULL,NULL),(38,'Neutral Tone',18,NULL,NULL),(39,'Cool Tone',18,NULL,NULL),(40,'Cool Tone',18,NULL,NULL),(41,'Neutral Tone',19,NULL,NULL),(42,'Neutral Tone',19,NULL,NULL),(43,'Cool Tone',19,NULL,NULL),(44,'Neutral Tone',18,NULL,NULL),(45,'Neutral Tone',18,NULL,NULL),(46,'Neutral Tone',18,NULL,NULL),(47,'Neutral Tone',18,NULL,NULL),(48,'Warm Tone',18,NULL,NULL),(49,'Neutral Tone',18,NULL,NULL),(50,'Neutral Tone',18,NULL,NULL),(51,'Neutral Tone',18,NULL,NULL),(52,'Neutral Tone',18,NULL,NULL),(53,'Warm Tone',18,NULL,NULL),(54,'Warm Tone',18,NULL,NULL),(55,'Warm Tone',18,NULL,NULL),(56,'Neutral Tone',18,NULL,NULL),(57,'Neutral Tone',18,NULL,NULL),(58,'Neutral Tone',18,NULL,NULL),(59,'Neutral Tone',18,NULL,NULL),(60,'Neutral Tone',18,NULL,NULL),(61,'Warm Tone',18,NULL,NULL),(62,'Warm Tone',18,NULL,NULL),(63,'Neutral Tone',18,NULL,NULL),(64,'Neutral Tone',18,NULL,NULL),(65,'Cool Tone',18,NULL,NULL),(66,'Cool Tone',18,NULL,NULL),(67,'Cool Tone',18,NULL,NULL),(68,'Warm Tone',18,NULL,NULL),(69,'Neutral Tone',18,NULL,NULL),(70,'Neutral Tone',18,NULL,NULL),(71,'Neutral Tone',18,NULL,NULL),(72,'Cool Tone',18,NULL,NULL),(73,'Neutral Tone',18,NULL,NULL),(74,'Cool Tone',18,NULL,NULL),(75,'Warm Tone',18,NULL,NULL),(76,'Warm Tone',18,NULL,NULL),(77,'Cool Tone',18,NULL,NULL),(78,'Cool Tone',18,NULL,NULL),(79,'Cool Tone',18,NULL,NULL),(80,'Cool Tone',18,NULL,NULL),(81,'Cool Tone',18,NULL,NULL),(82,'Cool Tone',18,NULL,NULL),(83,'Cool Tone',18,NULL,NULL),(84,'Cool Tone',18,NULL,NULL),(85,'Cool Tone',18,NULL,NULL),(86,'Cool Tone',18,NULL,NULL),(87,'Warm Tone',18,NULL,NULL),(88,'Warm Tone',18,NULL,NULL),(89,'Neutral Tone',18,NULL,NULL),(90,'Neutral Tone',18,NULL,NULL),(91,'Neutral Tone',18,NULL,NULL),(92,'Cool Tone',18,NULL,NULL),(93,'Cool Tone',18,NULL,NULL),(94,'Cool Tone',18,NULL,NULL);
/*!40000 ALTER TABLE `skintoneanalysis` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `thaicelebrities`
--

DROP TABLE IF EXISTS `thaicelebrities`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `thaicelebrities` (
  `ThaiCelebrities_ID` int NOT NULL AUTO_INCREMENT,
  `ThaiCelebrities_name` varchar(255) NOT NULL,
  PRIMARY KEY (`ThaiCelebrities_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=101 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `thaicelebrities`
--

LOCK TABLES `thaicelebrities` WRITE;
/*!40000 ALTER TABLE `thaicelebrities` DISABLE KEYS */;
INSERT INTO `thaicelebrities` VALUES (1,'ซุปเปอร์บอน'),(2,'กลัฟ คณาวุฒิ'),(3,'กวาง อาริศา หอมกรุ่น'),(4,'ก้อย อรัชพร'),(5,'เก๋ไก๋ สไลเดอร์'),(6,'จ๊ะ นงมณี'),(7,'เจมส์ จิรายุ'),(8,'ชนาธิป สรงกระสินธ์'),(9,'เต๋อ ฉันทวิชช์'),(10,'ญาญ่า อุรัสยา'),(11,'เบลล่า ราณี'),(12,'เบสท์ คำสิงห์'),(13,'โบว์ กัญญารัตน์'),(14,'ใบเฟิร์น พิมพ์ชนก'),(15,'ปอป้อ ทรัพย์สิรี แต้รัตนชัย'),(16,'ฝ้าย 4EVE'),(17,'มาเบล PIXXE'),(18,'มาริโอ้ เมาเร่อ'),(19,'ลิซ่า'),(20,'ลำไย ไหทองคำ'),(21,'สไปร์ท SPD'),(22,'โอบ นิธิ วิวรรธนวรางค์'),(23,'ใหม่ดาวิกา'),(24,'อั้ม พัชราภา'),(25,'อิงโกะ PiXXiE'),(26,'โอปป้าทัชชี่'),(27,'ฮาร์ท ชุติวัฒน์ จันเคน'),(28,'เอวา'),(29,'อาจุมม่า'),(30,'วี วิโอเลต วอเทียร์'),(31,'วิน เมธวิน'),(32,'มิ้น ชาลิดา'),(33,'มายเมทเนท'),(34,'ฟาง-ธนันต์ธรญ์'),(35,'แพนเค้ก เขมนิจ'),(36,'นิกกี้ นฉัตร'),(37,'แต้ว ณฐพร'),(38,'ต้าเหนิง กัญญาวีร์ สองเมือง'),(39,'เจ้าขุน'),(40,'จิดาภา แช่มช้อย'),(41,'เจ้านาย วรรธนะสิน'),(42,'โดนัท ภัทรพลฒ์ เดชพงษ์วรานนท์'),(43,'เก้า สุภัสสรา'),(44,'ขวัญ อุษามณี'),(45,'คริษฐา สังสะโอภาส'),(46,'คริส หอวัง'),(47,'ครีมไลค์'),(48,'คารีสา สปริงเก็ตต์'),(49,'บิว วราภรณ์'),(50,'คิมเบอร์ลี่'),(51,'บอส ชนกันต์'),(52,'จินวุค คิม'),(53,'เจมี่ จุฑาพิชญ์'),(54,'เจแปน ภาณุพรรณ จันทนะวงษ์'),(55,'แจน พลอยชมพู'),(56,'ซ้อการ์ด'),(57,'ฐิสา วริฏฐิสา'),(58,'ณัฐทิชา จันทรวารีเลขา'),(59,'ณัฐรุจา ชุติวรรณโสภณ'),(60,'ณิชาภัทร ฉัตรชัยพลรัตน์'),(61,'บูม กฤติน'),(62,'มิกค์ ทองระย้า'),(63,'แบงค์ปิ'),(64,'เจเจ ชยกร'),(65,'โบว์ เมลดา'),(66,'ไบร์ท วชิรวิชญ์ ชีวอารี'),(67,'ปู ไปรยา'),(68,'พลอย หอวัง'),(69,'ปูเป้ เกศรินทร์'),(70,'พิ้งกี้ สาวิกา'),(71,'มิลิน ดอกเทียน'),(72,'พิมรี่พาย'),(73,'อแมนด้า ออบดัม'),(74,'ภีม วสุพล พรพนานุรักษ์'),(75,'อาเล็ก ธีรเดช เมธาวรายุทธ'),(76,'โอ๊ต ปราโมทย์'),(77,'ยูโร ยศวรรธน์'),(78,'สรยุทธ สุทัศนะจินดา'),(79,'สุภโชค สารชาติ'),(80,'นิว พีรดนย์'),(81,'แบงค์ ศุภณัฏฐ์'),(82,'ตะวันฉาย'),(83,'ดัง ณัฎฐ์ฐชัย'),(84,'ซุง ศตาวิน นาคทองเพชร'),(85,'ซิม คิวเท'),(86,'ชาริล ชับปุยส์'),(87,'ไทย ชญานนท์ ภาคฐิน'),(88,'มาร์ค กฤษณ์ กัญจนาทิพย์'),(89,'พีระกฤตย์ พชรบุณยเกียรติ'),(90,'กัน อรรถพันธ์'),(91,'โอห์ม ฐิติวัฒน์'),(92,'กาย ศิวกร'),(93,'ขุนพล ปองพล ปัญญามิตร'),(94,'เคน - ภูภูมิ พงศ์ภาณุ'),(95,'ฮง พิเชฐพงศ์'),(96,'ซี เดชชาติ'),(97,'เซียนหรั่ง'),(98,'เดรก สัตบุตร'),(99,'เอส ศุภ'),(100,'ไมค์ ภัทรเดช');
/*!40000 ALTER TABLE `thaicelebrities` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `Users_ID` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `Role_ID` int DEFAULT NULL,
  PRIMARY KEY (`Users_ID`),
  KEY `Role_ID` (`Role_ID`),
  CONSTRAINT `fk_users_role` FOREIGN KEY (`Role_ID`) REFERENCES `role` (`Role_ID`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `users_ibfk_1` FOREIGN KEY (`Role_ID`) REFERENCES `role` (`Role_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'admin','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(2,'pichai','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(17,'Test','$2b$12$WChZnogLJd8ZJArPkVBriu93FDp5tW1s.oNWaEVlJYcJ2L5KEM/Fe',1),(18,'Nack','$2b$12$gAl8n6B8AXeYzcMva2lFJu54DbIrapNgWYi1anXcdgo1zA5.Y7/re',1),(19,'duangjai','$2b$12$ao3JXzw7UwK4Z/trDUsV7ezLqHZa7Hwp8sZxyD40KOK8syYFx6/ie',1);
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Temporary view structure for view `v_feedback_stats`
--

DROP TABLE IF EXISTS `v_feedback_stats`;
/*!50001 DROP VIEW IF EXISTS `v_feedback_stats`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `v_feedback_stats` AS SELECT 
 1 AS `CosmeticID`,
 1 AS `total_reviews`,
 1 AS `liked`,
 1 AS `disliked`*/;
SET character_set_client = @saved_cs_client;

--
-- Dumping routines for database 'db_miniprojectfinal'
--

--
-- Final view structure for view `v_feedback_stats`
--

/*!50001 DROP VIEW IF EXISTS `v_feedback_stats`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `v_feedback_stats` AS select `feedback`.`CosmeticID` AS `CosmeticID`,count(0) AS `total_reviews`,sum((case when (`feedback`.`Rating` >= 4) then 1 else 0 end)) AS `liked`,sum((case when (`feedback`.`Rating` <= 2) then 1 else 0 end)) AS `disliked` from `feedback` where (`feedback`.`CosmeticID` is not null) group by `feedback`.`CosmeticID` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-09-25 23:27:19
