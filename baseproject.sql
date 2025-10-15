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
-- Table structure for table `activity_log`
--

DROP TABLE IF EXISTS `activity_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `activity_log` (
  `log_id` int NOT NULL AUTO_INCREMENT,
  `operator_id` int DEFAULT NULL,
  `action_type` varchar(50) NOT NULL,
  `description` varchar(255) NOT NULL,
  `status` varchar(50) NOT NULL,
  `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`log_id`),
  KEY `operator_id` (`operator_id`),
  CONSTRAINT `activity_log_ibfk_1` FOREIGN KEY (`operator_id`) REFERENCES `users` (`Users_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=106 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `activity_log`
--

LOCK TABLES `activity_log` WRITE;
/*!40000 ALTER TABLE `activity_log` DISABLE KEYS */;
INSERT INTO `activity_log` VALUES (1,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 20:34:24'),(2,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 20:38:51'),(3,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:09:21'),(4,1,'FEEDBACK_UPDATE','เปลี่ยนสถานะ Feedback ID: #1 เป็น \'ยังไม่ได้อ่าน\'','แก้ไข','2025-08-14 21:09:30'),(5,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:10:32'),(6,1,'FEEDBACK_UPDATE','เปลี่ยนสถานะ Feedback ID: #2 เป็น \'แก้ไขแล้ว\'','แก้ไข','2025-08-14 21:10:56'),(7,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:19:04'),(8,17,'LOGIN_NO_AUTH','ผู้ใช้ \'test\' พยายามเข้าสู่ระบบ แต่ไม่มีสิทธิ์ (Role: user)','ไม่สำเร็จ','2025-08-14 21:28:13'),(9,17,'LOGIN_NO_AUTH','ผู้ใช้ \'test\' พยายามเข้าสู่ระบบ แต่ไม่มีสิทธิ์ (Role: user)','ไม่สำเร็จ','2025-08-14 21:28:14'),(10,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:28:20'),(11,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:37:29'),(12,17,'LOGIN_NO_AUTH','ผู้ใช้ \'test\' พยายามเข้าสู่ระบบ แต่ไม่มีสิทธิ์ (Role: user)','ไม่สำเร็จ','2025-08-14 21:39:12'),(13,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-14 21:39:16'),(14,17,'LOGIN_NO_AUTH','ผู้ใช้ \'test\' พยายามเข้าสู่ระบบ แต่ไม่มีสิทธิ์ (Role: user)','ไม่สำเร็จ','2025-08-15 03:16:47'),(15,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-15 03:17:08'),(16,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-15 04:55:51'),(17,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-15 05:01:10'),(18,1,'FEEDBACK_UPDATE','เปลี่ยนสถานะ Feedback ID: #1 เป็น \'อ่านแล้ว\'','แก้ไข','2025-08-15 05:05:50'),(19,1,'FEEDBACK_UPDATE','เปลี่ยนสถานะ Feedback ID: #1 เป็น \'ยังไม่ได้อ่าน\'','แก้ไข','2025-08-15 05:06:28'),(20,1,'FEEDBACK_UPDATE','เปลี่ยนสถานะ Feedback ID: #2 เป็น \'ยังไม่ได้อ่าน\'','แก้ไข','2025-08-15 05:06:31'),(21,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-18 05:53:10'),(22,17,'LOGIN_NO_AUTH','ผู้ใช้ \'test\' พยายามเข้าสู่ระบบ แต่ไม่มีสิทธิ์ (Role: user)','ไม่สำเร็จ','2025-08-18 05:53:37'),(23,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-18 05:53:50'),(24,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-18 07:55:36'),(25,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-08-18 08:11:37'),(26,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-09-24 20:42:29'),(27,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-09-25 17:46:08'),(28,1,'LOOK_CREATE','เพิ่มลุคใหม่: \'Test2\'','เพิ่ม','2025-09-25 17:47:32'),(29,1,'LOOK_DELETE','ลบลุค: \'Test2\'','ลบ','2025-09-25 17:47:47'),(30,1,'LOOK_CREATE','เพิ่มลุคใหม่: \'Test3\'','เพิ่ม','2025-09-25 17:50:01'),(31,1,'LOOK_DELETE','ลบลุค: \'Test3\'','ลบ','2025-09-25 17:50:15'),(32,1,'LOOK_UPDATE','แก้ไขข้อมูลลุค ID #11: \'Test\'','แก้ไข','2025-09-25 17:50:27'),(33,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-09-26 03:06:15'),(34,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-07 06:51:16'),(35,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 09:10:43'),(36,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Lustreglass Lipstick\'','เพิ่ม','2025-10-09 09:18:27'),(37,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 10:38:50'),(38,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Test\'','เพิ่ม','2025-10-09 10:39:34'),(39,1,'PRODUCT_DELETE','ลบสินค้า: \'Test\'','ลบ','2025-10-09 10:39:50'),(40,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 12:13:21'),(41,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 12:16:20'),(42,1,'PRODUCT_DELETE','ลบสินค้า: \'Lustreglass Lipstick\'','ลบ','2025-10-09 12:19:23'),(43,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 12:58:26'),(44,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 13:00:17'),(45,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Twe\'','เพิ่ม','2025-10-09 13:01:06'),(46,1,'PRODUCT_DELETE','ลบสินค้า: \'Twe\'','ลบ','2025-10-09 13:01:09'),(47,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Test\'','เพิ่ม','2025-10-09 13:01:31'),(48,1,'PRODUCT_DELETE','ลบสินค้า: \'Test\'','ลบ','2025-10-09 13:01:36'),(49,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'asd\'','เพิ่ม','2025-10-09 13:03:05'),(50,1,'PRODUCT_DELETE','ลบสินค้า: \'asd\'','ลบ','2025-10-09 13:03:12'),(51,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 13:07:12'),(52,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 13:07:59'),(53,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #34: \'Chocolate Soleil Matte Bronzer\'','แก้ไข','2025-10-09 13:10:07'),(54,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #33: \'Soft Matte Lip Cream\'','แก้ไข','2025-10-09 13:11:25'),(55,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #32: \'Born This Way Super Coverage Concealer\'','แก้ไข','2025-10-09 13:12:20'),(56,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #31: \'Radiant Creamy Concealer\'','แก้ไข','2025-10-09 13:13:23'),(57,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 13:13:58'),(58,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 13:15:38'),(59,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #34: \'Chocolate Soleil Matte Bronzer\'','แก้ไข','2025-10-09 13:16:09'),(60,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-09 14:22:31'),(61,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 14:22:49'),(62,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #34: \'Chocolate Soleil Matte Bronzer\'','แก้ไข','2025-10-09 14:23:14'),(63,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #33: \'Soft Matte Lip Cream\'','แก้ไข','2025-10-09 14:23:38'),(64,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #32: \'Born This Way Super Coverage Concealer\'','แก้ไข','2025-10-09 14:23:56'),(65,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #31: \'Radiant Creamy Concealer\'','แก้ไข','2025-10-09 14:24:29'),(66,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #30: \'Cream Lip Stain\'','แก้ไข','2025-10-09 14:24:44'),(67,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #29: \'Brow Wiz\'','แก้ไข','2025-10-09 14:25:02'),(68,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #21: \'Pro Filt\'r Soft Matte Longwear Foundation\'','แก้ไข','2025-10-09 14:25:25'),(69,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #22: \'Retro Matte Lipstick\'','แก้ไข','2025-10-09 14:25:39'),(70,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #23: \'Blush\'','แก้ไข','2025-10-09 14:25:58'),(71,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #24: \'Fit Me Concealer\'','แก้ไข','2025-10-09 14:26:17'),(72,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #28: \'Matte Lip Kit\'','แก้ไข','2025-10-09 14:26:36'),(73,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #27: \'Cookie Powder Highlighter\'','แก้ไข','2025-10-09 14:26:50'),(74,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #26: \'Dior Forever Skin Glow Foundation\'','แก้ไข','2025-10-09 14:27:03'),(75,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #25: \'Infallible 24H Fresh Wear Foundation\'','แก้ไข','2025-10-09 14:27:19'),(76,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #21: \'Pro Filt\'r Soft Matte Longwear Foundation\'','แก้ไข','2025-10-09 14:28:38'),(77,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #22: \'Retro Matte Lipstick\'','แก้ไข','2025-10-09 14:29:04'),(78,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #23: \'Blush\'','แก้ไข','2025-10-09 14:29:17'),(79,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #24: \'Fit Me Concealer\'','แก้ไข','2025-10-09 14:29:24'),(80,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #25: \'Infallible 24H Fresh Wear Foundation\'','แก้ไข','2025-10-09 14:29:30'),(81,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #26: \'Dior Forever Skin Glow Foundation\'','แก้ไข','2025-10-09 14:29:40'),(82,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #27: \'Cookie Powder Highlighter\'','แก้ไข','2025-10-09 14:30:27'),(83,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #28: \'Matte Lip Kit\'','แก้ไข','2025-10-09 14:30:41'),(84,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #29: \'Brow Wiz\'','แก้ไข','2025-10-09 14:31:24'),(85,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #30: \'Cream Lip Stain\'','แก้ไข','2025-10-09 14:31:50'),(86,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #31: \'Radiant Creamy Concealer\'','แก้ไข','2025-10-09 14:32:13'),(87,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #32: \'Born This Way Super Coverage Concealer\'','แก้ไข','2025-10-09 14:32:52'),(88,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #33: \'Soft Matte Lip Cream\'','แก้ไข','2025-10-09 14:33:06'),(89,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #34: \'Chocolate Soleil Matte Bronzer\'','แก้ไข','2025-10-09 14:33:13'),(90,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #35: \'Lip Glow Oil\'','แก้ไข','2025-10-09 14:33:32'),(91,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-10 02:53:16'),(92,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-10 04:12:23'),(93,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Super Stay Matte Ink® Liquid Lipstick\'','เพิ่ม','2025-10-10 04:32:52'),(94,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #41: \'Super Stay Matte Ink® Liquid Lipstick\'','แก้ไข','2025-10-10 04:33:50'),(95,1,'LOOK_CREATE','เพิ่มลุคใหม่: \'ธรรมชา\'','เพิ่ม','2025-10-10 04:49:05'),(96,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-11 09:37:54'),(97,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-11 09:40:34'),(98,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-15 12:18:18'),(99,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-15 12:23:51'),(100,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-15 13:00:46'),(101,1,'PRODUCT_CREATE','เพิ่มสินค้าใหม่: \'Test\'','เพิ่ม','2025-10-15 13:01:15'),(102,1,'PRODUCT_DELETE','ลบสินค้า: \'Test\'','ลบ','2025-10-15 13:01:35'),(103,1,'LOGIN_SUCCESS','Admin \'admin\' เข้าสู่ระบบ','สำเร็จ','2025-10-15 13:58:26'),(104,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #41: \'Super Stay Matte Ink® Liquid Lipstick\'','แก้ไข','2025-10-15 14:07:39'),(105,1,'PRODUCT_UPDATE','แก้ไขข้อมูลสินค้า ID #41: \'Super Stay Matte Ink® Liquid Lipstick\'','แก้ไข','2025-10-15 14:07:57');
/*!40000 ALTER TABLE `activity_log` ENABLE KEYS */;
UNLOCK TABLES;

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
) ENGINE=InnoDB AUTO_INCREMENT=194 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
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
  `suitableLookType` varchar(255) DEFAULT NULL,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `HexCode` varchar(7) DEFAULT NULL,
  `RGBCode` varchar(20) DEFAULT NULL,
  `Lab_L` decimal(6,2) DEFAULT NULL COMMENT 'CIELAB L*',
  `Lab_a` decimal(6,2) DEFAULT NULL COMMENT 'CIELAB a*',
  `Lab_b` decimal(6,2) DEFAULT NULL COMMENT 'CIELAB b*',
  PRIMARY KEY (`CosmeticID`),
  KEY `BrandID` (`BrandID`),
  CONSTRAINT `cosmetics_ibfk_1` FOREIGN KEY (`BrandID`) REFERENCES `brand` (`brandID`),
  CONSTRAINT `fk_cosmetics_brand` FOREIGN KEY (`BrandID`) REFERENCES `brand` (`brandID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=43 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cosmetics`
--

LOCK TABLES `cosmetics` WRITE;
/*!40000 ALTER TABLE `cosmetics` DISABLE KEYS */;
INSERT INTO `cosmetics` VALUES (21,'Pro Filt\'r Soft Matte Longwear Foundation','240','','Foundation','',1600.00,'/images/1760019922821-282731158.jpg','https://www.sephora.co.th/products/fenty-beauty-pro-filtr-soft-matte-longwear-foundation/v/100',5,'Medium','สายฝอ,Everyday Glam,งานกลางคืน,งานรับปริญญา','2025-10-08 07:06:40','2025-10-09 14:28:38','#e1b694','225, 182, 148',77.33,12.59,23.54),(22,'Retro Matte Lipstick','707','Ruby Woo','Lipstick','รูบี้วู้ VIVID BLUE-RED แบบแมตต์สดใสมาก รายละเอียดทั้งหมดนี้เป็นแบบเต็ม M·A·C Lipstick - ผลิตภัณฑ์อันเป็นสัญลักษณ์ที่ทำให้ M·A·C มีชื่อเสียง สูตรติดทนนานนี้ให้ผลลัพธ์สีที่เข้มข้นและผิวด้านที่สมบูรณ์ การเรียกร้องและสิทธิประโยชน์ที่สำคัญ สวมใส่ได้นานแปดชั่วโมง ไม่มีขน ป้องกันการซีดจางป้องกันการซีดจาง การใช้งาน ทาลงบนริมฝีปากโดยตรงจากกระสุนลิปสติกหรือใช้แปรง 316 เพื่อความแม่นยำยิ่งขึ้น',900.00,'/images/1760019934666-125751755.jpeg','https://shopee.co.th/%E2%97%91MAC-Cosmetics-Retro-Matte-Lipstick-Ruby-Woo-i.1032037872.18681789672',2,'Universal','สายฝอ,งานกลางคืน','2025-10-08 07:08:29','2025-10-09 14:29:04','#9b111e','155, 17, 30',33.51,53.68,33.50),(23,'Blush',NULL,'Blush','Blush','NARS บลัชออน Blush บลัชที่ดีที่สุดสำหรับแต่งหน้า มอบสีสัน เพื่อการแต่งแต้มความมั่นใจให้กับคุณ Nars ขึ้นชื่อในเรื่องเม็ดสีที่เด่นชัด ให้สีสันระเรื่อแก่พวงแก้ม สีแก้มของ Nars ไม่ทึบแสงเหมือนปัดแก้มในยุคเก่า โดยใช้เม็ดสีที่โปร่งใส ผลลัพธ์คือสีสวยที่นุ่มนวล ดุจดั่งสีสันจากธรรมชาติ ที่สำคัญได้รับการกล่าวขานว่า \"คงความสดชัดของสีจริง ติดทนนานกว่าบลัชทั่วไป\" คุณสมบัติ เม็ดสีชัดตั้งแต่ครั้งแรกที่ทา ให้ความรู้สึกธรรมชาติเหมือนมีเลือดฝาด',1300.00,'/images/1760019954934-942770672.jpg','https://shopee.co.th/NARS-Blush-ORGASM-4.8g.-i.48297855.1703679411',7,'Fair, Medium','ธรรมชาติ,Everyday Glam','2025-10-08 07:09:15','2025-10-09 14:29:17','#e7a391','231, 163, 145',73.38,24.30,19.98),(24,'Fit Me Concealer','20','Sand','Concealer','ฟิต มี คอนซีลเลอร์ ปกปิดรอยคล้ำใต้ตา รอยดำ รอบแดง บนใบหน้าให้เรียบเนียน',249.00,'/images/1760019975285-456204253.jpg','https://shopee.co.th/-A052-%E0%B9%81%E0%B8%97%E0%B9%89-100-Maybelline-fit-me-concealer-i.6281919.5067701764?sp_atk=6646c194-9827-47ef-a10c-b43f5dad5c5c&xptdk=6646c194-9827-47ef-a10c-b43f5dad5c5c',3,'Fair, Medium','ธรรมชาติ,Everyday Glam','2025-10-08 07:11:42','2025-10-09 14:29:24','#f3d1b3','243, 209, 179',86.20,9.05,19.58),(25,'Infallible 24H Fresh Wear Foundation','130','True Beige','Foundation','แป้งไฮบริด นวัตกรรมการปกปิดเรียบเนียนแบบรองพื้น แต่สัมผัสบางเบากลืนผิวแบบแป้ง ด้วยเทคโนโลยี HYBRID LIQUID-IN-POWDER เปลี่ยนอณูแป้งเนียนละเอียดให้ผสานกลืนผิวเป็นหนึ่งเดียว เพื่อผิวสวยดูเป็นธรรมชาติ แต่ปกปิดเรียบเนียนไร้ที่ติ คุมมัน ติดทนตลอดวัน ทนน้ำ ทนเหงื่อ ไม่เป็นคราบ หน้าไม่ดรอประหว่างวัน มาพร้อมแพ็กเกจที่มีพัพฟ์และกระจกในตัว มีให้เลือกถึง 6 เฉดเพื่อสีผิวคนไทย',449.00,'/images/1760020038238-357855299.jpg','https://www.tops.co.th/en/loreal-paris-infallible-24h-fresh-wear-foundation-no130-30ml-6902395719380',4,'Fair, Medium','ธรรมชาติ,Everyday Glam','2025-10-08 07:11:42','2025-10-09 14:29:30','#f0c9a5','240, 201, 165',83.73,10.51,23.47),(26,'Dior Forever Skin Glow Foundation','2N','Neutral','Foundation','',2500.00,'/images/1760020021198-20575422.jpg','https://www.lazada.co.th/products/dior-forever-skin-glow-foundation-spf35-pa-30ml-24-i4456695820-s17936478602.html?',1,'Fair, Medium','ธรรมชาติ,Everyday Glam','2025-10-08 07:11:42','2025-10-09 14:29:40','#e3b997','227, 185, 151',78.31,12.14,23.31),(27,'Cookie Powder Highlighter',NULL,'Cookie','Highlighter','',1400.00,'/images/1760020008716-548744242.jpg','https://www.sephora.co.th/products/benefit-cosmetics-cookie-golden-pearl-highlighter/v/8g',10,'Fair, Medium, Brown','Everyday Glam,งานกลางคืน','2025-10-08 07:19:57','2025-10-09 14:30:27','#f6d5a1','246, 213, 161',87.18,6.59,30.25),(28,'Matte Lip Kit',NULL,'Kristen','Lip Kit','',1350.00,'/images/1760019993864-684173008.jpg','https://www.lazada.co.th/products/anastasia-beverly-hills-brow-wiz-i4769305159-s19675092130.html?',8,'Universal','สายฝอ,Everyday Glam','2025-10-08 07:19:57','2025-10-09 14:30:41','#c06362','192, 99, 98',53.40,37.92,18.52),(29,'Brow Wiz',NULL,'Medium Brown','Eyebrow Pencil','',950.00,'/images/1760019899760-274410623.jpg','https://www.lazada.co.th/products/anastasia-beverly-hills-brow-wiz-i4769305159-s19675092130.html?',9,'Universal','ธรรมชาติ,สายเกาหลี,สายฝอ,Everyday Glam,งานรับปริญญา','2025-10-08 07:19:57','2025-10-09 14:31:24','#381f1a','56, 31, 26',15.09,11.99,8.83),(30,'Cream Lip Stain','1','Always Red','Liquid Lipstick','',520.00,'/images/1760019881754-835794094.jpg','https://www.lazada.co.th/products/sephora-collection-new-cream-lip-stain-i5293024994-s22540217011.html?',6,'Universal','สายฝอ,งานกลางคืน','2025-10-08 07:19:57','2025-10-09 14:31:50','#a01a1a','160, 26, 26',35.25,53.12,37.88),(31,'Radiant Creamy Concealer',NULL,'Custard','Concealer','คอนซีลเลอร์เนื้อครีมอณูเม็ดสีเข้มข้น ให้การปกปิดริ้วรอยและจุดด่างดำได้อย่างแนบเนียน พร้อมคุณสมบัติในการมอบความชุ่มชื่นสู่ผิว เผยผิวเปล่งปลั่ง เรียบเนียนอย่างเป็นธรรมชาติ',1200.00,'/images/1760019865314-713507610.jpg','https://www.lazada.co.th/products/nars-radiant-creamy-concealer-6ml-i752346317-s1437806662.html?',7,'Fair, Medium','ธรรมชาติ,Everyday Glam','2025-10-08 07:19:57','2025-10-09 14:32:13','#e1b48f','225, 180, 143',76.76,13.16,25.44),(32,'Born This Way Super Coverage Concealer',NULL,'Nude','Concealer','เป็นคอนซีลเลอร์แบบปกปิดเต็มกริบ (full coverage)\n\nรอยหนักๆ เอาอยู่',1200.00,'/images/1760019833706-399964227.jpg','https://shopee.co.th/%E0%B9%81%E0%B8%97%E0%B9%89%F0%9F%92%AF-%E0%B9%80%E0%B8%84%E0%B8%B2%E0%B8%99%E0%B9%8C%E0%B9%80%E0%B8%95%E0%B8%AD%E0%B8%A3%E0%B9%8C%E0%B8%AD%E0%B9%80%E0%B8%A1%E0%B8%A3%E0%B8%B4%E0%B8%81%E0%B8%B2-%E0%B9%80%E0%B8%8A%E0%B9%87%E0%B8%84%E0%B9%83%E0%B8%9A%E0%B9%80%E0%B8%AA%E0%B8%A3%E0%B9%87%E0%B8%88%E0%B9%84%E0%B8%94%E0%B9%89-Too-Faced-Born-This-Way-Super-Coverage-Concealer-i.2283759.3971954006',55,'Fair, Medium','ธรรมชาติ,สายฝอ,Everyday Glam,งานกลางคืน,งานรับปริญญา','2025-10-08 07:19:57','2025-10-09 14:32:52','#e7b99b','231, 185, 155',78.78,14.07,21.90),(33,'Soft Matte Lip Cream',NULL,'Stockholm','Liquid Lipstick','NYX Soft Matte Lip Cream ลิปสติกเนื้อบางเบา สุดฮิตจาก Tiktok',299.00,'/images/1760019816123-460077094.jpg','https://www.lazada.co.th/products/nyx-soft-matte-lip-cream-tiktok-i4220971190-s16630100990.html?',9,'Fair, Medium','ธรรมชาติ,สายฝอ','2025-10-08 07:19:57','2025-10-09 14:33:06','#c68772','198, 135, 114',62.47,23.01,21.60),(34,'Chocolate Soleil Matte Bronzer',NULL,'Medium','Bronzer','',1400.00,'/images/1760019791891-52826590.jpg','https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-Too-faced-Chocolate-Soleil-Matte-Bronzer-i.1300084.505593354',55,'Medium, Brown','ธรรมชาติ,สายฝอ','2025-10-08 07:19:57','2025-10-09 14:33:13','#a76d4b','167, 109, 75',51.82,21.06,28.86),(35,'Lip Glow Oil','1','Pink','Lip Oil','Dior Addict Lip Glow Oil\n\nลิปออยล์บำรุงริมฝีปาก ประกายงาม เนื้อสบาย ไม่เหนียว',1600.00,'/images/1760019764928-108860036.jpg','https://shopee.co.th/-%E0%B8%9E%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%AA%E0%B9%88%E0%B8%87-Dior-Addict-Lip-Glow-Oil-%E0%B8%A5%E0%B8%B4%E0%B8%9B%E0%B8%AD%E0%B8%AD%E0%B8%A2%E0%B8%A5%E0%B9%8C-%E0%B8%94%E0%B8%B4%E0%B8%AD%E0%B8%AD%E0%B8%A3%E0%B9%8C-i.147535289.14099430262?sp_atk=4e2a5776-6a22-414b-8e73-8cdca33ee078&xptdk=4e2a5776-6a22-414b-8e73-8cdca33ee078',1,'Universal','ธรรมชาติ,สายเกาหลี','2025-10-08 07:19:57','2025-10-09 14:33:32','#f4b0c4','244, 176, 196',78.81,27.64,0.19),(41,'Super Stay Matte Ink® Liquid Lipstick',NULL,'Exhilarator','Lipstick','',384.00,'/images/1760070694222-59497120.jpg','https://www.maybelline.com/lip-makeup/lipstick/superstay-matte-ink-liquid-lipstick?variant=Exhilarator',3,'Fair','สายฝอ','2025-10-10 04:32:52','2025-10-15 14:07:57','#2596be','37, 150, 190',57.46,-21.06,-30.14);
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
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `feedback`
--

LOCK TABLES `feedback` WRITE;
/*!40000 ALTER TABLE `feedback` DISABLE KEYS */;
INSERT INTO `feedback` VALUES (1,'แอปดีมากเลยค่ะ ชอบสุดๆ!',5,'2025-08-14',17,NULL),(2,'TEst',3,'2025-08-14',17,NULL),(3,'asd',4,'2025-09-22',18,NULL),(4,'good',5,'2025-09-24',18,NULL),(5,'สนใจฝึกงานในตำแหน่ง Front-end Developer (Internship)',5,'2025-10-08',18,NULL),(6,'very good',5,'2025-10-10',19,NULL);
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
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `makeuplook`
--

LOCK TABLES `makeuplook` WRITE;
/*!40000 ALTER TABLE `makeuplook` DISABLE KEYS */;
INSERT INTO `makeuplook` VALUES (1,'ธรรมชาติ','Everyday Looks','ธรรมชาติใสๆ'),(2,'สายเกาหลี','Cultural Styles','เน้นผิวฉ่ำวาว อายไลเนอร์บางเบา สีปากสดใส'),(3,'สายฝอ','Cultural Styles','เน้นโครงหน้าชัด คอนทัวร์หนัก อายแชโดว์คมเข้ม'),(4,'สโมคกี้อายส์','Party Looks','เน้นดวงตาโดดเด่น ด้วยอายแชโดว์สีเข้ม'),(5,'Everyday Glam','Everyday Looks','ลุคที่ดูแต่งหน้า แต่ยังคงความเบาและสดใส เหมาะกับทุกวัน'),(6,'งานกลางคืน','Party Looks','ลุคสำหรับออกงานกลางคืน หรูหราและโดดเด่น'),(7,'วินเทจ','Thematic Looks','สไตล์การแต่งหน้าย้อนยุค เช่น ยุค 60s, 70s'),(8,'เทรนดี้','Seasonal Looks','ลุคที่กำลังอินเทรนด์ในปัจจุบัน'),(9,'งานรับปริญญา','Special Occasion','ลุคที่สุภาพ แต่ยังคงความสวยงาม เหมาะกับวันสำคัญ'),(10,'แนวพังก์','Alternative Styles','ลุคที่เน้นความขบถและมีเอกลักษณ์เฉพาะตัว'),(11,'ธรรมชา','Everyday Looks','');
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
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `recommendedcolorpalettes`
--

LOCK TABLES `recommendedcolorpalettes` WRITE;
/*!40000 ALTER TABLE `recommendedcolorpalettes` DISABLE KEYS */;
INSERT INTO `recommendedcolorpalettes` VALUES (1,'Fair','Fair','medium.jpg','ตารางสีสำหรับผิวโทนสว่าง','2025-07-31 14:17:12','2025-10-08 14:49:23'),(2,'Medium','Medium','fair.jpg','ตารางสีสำหรับผิวโทนกลาง','2025-07-31 14:17:12','2025-10-08 14:49:23'),(3,'Brown','Brown','brown.jpg','ตารางสีสำหรับผิวโทนน้ำตาล','2025-07-31 14:17:12','2025-10-08 14:49:23'),(4,'Deep dark','Deep dark','deep.jpg','ตารางสีสำหรับผิวโทนเข้ม','2025-10-08 14:48:29','2025-10-08 15:04:52');
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
) ENGINE=InnoDB AUTO_INCREMENT=103 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `similarity`
--

LOCK TABLES `similarity` WRITE;
/*!40000 ALTER TABLE `similarity` DISABLE KEYS */;
INSERT INTO `similarity` VALUES (67,'2025-02-25',35.26,81,17),(68,'2025-02-25',35.26,81,17),(69,'2025-02-25',29.50,36,17),(70,'2025-02-25',35.26,81,17),(71,'2025-02-25',33.71,67,17),(72,'2025-03-05',29.88,26,17),(73,'2025-03-05',30.59,26,17),(74,'2025-03-05',30.34,26,17),(75,'2025-03-05',30.36,26,17),(76,'2025-07-30',57.98,47,17),(77,'2025-07-31',46.59,59,17),(78,'2025-07-31',46.59,59,17),(79,'2025-07-31',46.14,59,17),(80,'2025-07-31',45.28,59,17),(81,'2025-07-31',44.86,59,17),(82,'2025-08-12',46.68,59,17),(83,'2025-08-12',46.68,59,17),(84,'2025-08-12',46.49,59,17),(85,'2025-08-12',45.02,59,17),(86,'2025-08-12',44.70,59,17),(87,'2025-08-12',46.73,59,17),(88,'2025-08-14',56.92,48,17),(89,'2025-08-15',55.70,91,18),(90,'2025-08-15',63.87,17,18),(91,'2025-08-15',68.60,17,18),(92,'2025-08-15',68.60,17,17),(93,'2025-08-15',68.60,17,17),(94,'2025-08-15',68.60,17,17),(95,'2025-08-15',68.60,17,17),(96,'2025-08-15',68.60,17,17),(97,'2025-08-15',64.82,10,17),(98,'2025-08-18',57.15,61,18),(99,'2025-08-18',63.87,17,18),(100,'2025-08-18',55.39,38,19),(101,'2025-08-18',60.81,63,19),(102,'2025-10-10',48.93,75,18);
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
  `L_star` decimal(6,2) DEFAULT NULL COMMENT 'CIELAB L* ของผิวผู้ใช้',
  `b_star` decimal(6,2) DEFAULT NULL COMMENT 'CIELAB b* ของผิวผู้ใช้',
  `ITA_Deg` decimal(6,2) DEFAULT NULL COMMENT 'ค่ามุม ITA ของผิวผู้ใช้',
  PRIMARY KEY (`SkinToneAnalysisID`),
  KEY `Users_ID` (`Users_ID`),
  CONSTRAINT `fk_skin_user` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `skintoneanalysis_ibfk_1` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=216 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `skintoneanalysis`
--

LOCK TABLES `skintoneanalysis` WRITE;
/*!40000 ALTER TABLE `skintoneanalysis` DISABLE KEYS */;
INSERT INTO `skintoneanalysis` VALUES (96,'Medium',18,NULL,90,NULL,NULL,NULL),(97,'Fair',18,NULL,100,NULL,NULL,NULL),(98,'Fair',18,NULL,100,NULL,NULL,NULL),(99,'Fair',18,NULL,100,NULL,NULL,NULL),(100,'Deep Dark',18,NULL,30,NULL,NULL,NULL),(101,'Fair',18,NULL,100,NULL,NULL,NULL),(102,'Medium',18,NULL,90,NULL,NULL,NULL),(103,'Deep',18,NULL,100,NULL,NULL,NULL),(104,'Deep',18,NULL,100,NULL,NULL,NULL),(105,'Fair',18,NULL,100,NULL,NULL,NULL),(106,'Fair',18,NULL,100,NULL,NULL,NULL),(107,'Fair',18,NULL,100,NULL,NULL,NULL),(108,'Fair',18,NULL,99,NULL,NULL,NULL),(109,'Fair',18,NULL,100,NULL,NULL,NULL),(110,'Fair',18,NULL,100,64.44,11.02,52.64),(111,'Fair',18,NULL,100,64.44,11.02,52.64),(112,'Fair',18,NULL,100,64.44,11.02,52.64),(113,'Fair',18,NULL,100,64.44,11.02,52.64),(114,'Fair',18,NULL,100,64.44,11.02,52.64),(115,'Medium',18,NULL,90,57.38,12.37,30.82),(116,'Fair',18,NULL,99,63.78,17.03,38.98),(117,'Fair',18,NULL,99,63.78,17.03,38.98),(118,'Medium',18,NULL,90,57.38,12.37,30.82),(119,'Fair',18,NULL,99,63.78,17.03,38.98),(120,'Fair',18,NULL,99,63.78,17.03,38.98),(121,'Fair',18,NULL,100,64.44,11.02,52.64),(122,'Fair',18,NULL,100,64.44,11.02,52.64),(123,'Fair',18,NULL,100,64.44,11.02,52.64),(124,'Fair',18,NULL,100,58.97,12.62,35.41),(125,'Fair',18,NULL,100,63.95,10.87,52.07),(126,'Fair',18,NULL,100,55.20,12.13,23.20),(127,'Fair',18,NULL,100,46.86,2.61,-50.30),(128,'Medium',18,NULL,90,57.38,12.37,30.82),(129,'Fair',18,NULL,99,63.78,17.03,38.98),(130,'Fair',18,NULL,99,63.78,17.03,38.98),(131,'Fair',18,NULL,99,63.78,17.03,38.98),(132,'Brown',18,NULL,97,55.75,11.94,25.74),(133,'Brown',18,NULL,97,55.75,11.94,25.74),(134,'Brown',18,NULL,97,55.75,11.94,25.74),(135,'Brown',18,NULL,97,55.75,11.94,25.74),(136,'Brown',18,NULL,97,55.75,11.94,25.74),(137,'Fair',18,NULL,100,82.42,16.08,63.62),(138,'Fair',18,NULL,100,82.42,16.08,63.62),(139,'Deep',18,NULL,100,35.42,9.85,-55.96),(140,'Fair',18,NULL,100,82.42,16.08,63.62),(141,'Fair',18,NULL,99,63.78,17.03,38.98),(142,'Deep',18,NULL,100,35.42,9.85,-55.96),(143,'Deep',18,NULL,100,35.42,9.85,-55.96),(144,'Brown',18,NULL,97,55.75,11.94,25.74),(145,'Brown',18,NULL,97,55.75,11.94,25.74),(146,'Fair',18,NULL,100,82.42,16.08,63.62),(147,'Fair',18,NULL,100,82.42,16.08,63.62),(148,'Fair',18,NULL,100,82.42,16.08,63.62),(149,'Fair',18,NULL,100,82.42,16.08,63.62),(150,'Brown',18,NULL,97,55.75,11.94,25.74),(151,'Fair',18,NULL,100,82.42,16.08,63.62),(152,'Fair',18,NULL,100,82.42,16.08,63.62),(153,'Fair',18,NULL,100,82.42,16.08,63.62),(154,'Fair',18,NULL,100,82.42,16.08,63.62),(155,'Fair',18,NULL,100,82.42,16.08,63.62),(156,'Fair',18,NULL,100,82.42,16.08,63.62),(157,'Fair',18,NULL,100,82.42,16.08,63.62),(158,'Deep',18,NULL,100,35.42,9.85,-55.96),(159,'Brown',18,NULL,97,55.75,11.94,25.74),(160,'Fair',18,NULL,100,82.42,16.08,63.62),(161,'Deep',18,NULL,100,35.42,9.85,-55.96),(162,'Deep',18,NULL,100,35.42,9.85,-55.96),(163,'Brown',18,NULL,97,55.75,11.94,25.74),(164,'Brown',18,NULL,97,55.75,11.94,25.74),(165,'Brown',18,NULL,97,55.75,11.94,25.74),(166,'Fair',18,NULL,100,82.42,16.08,63.62),(167,'Brown',18,NULL,97,55.75,11.94,25.74),(168,'Brown',18,NULL,97,55.75,11.94,25.74),(169,'Fair',18,NULL,100,82.42,16.08,63.62),(170,'Fair',18,NULL,100,82.42,16.08,63.62),(171,'Brown',18,NULL,72,47.30,5.80,-24.93),(172,'Brown',18,NULL,72,47.30,5.80,-24.93),(173,'Fair',19,NULL,99,48.90,12.18,-5.16),(174,'Medium',19,NULL,98,48.80,9.88,-6.95),(175,'Medium',19,NULL,85,46.33,11.11,-18.27),(176,'Fair',19,NULL,99,51.92,9.66,11.22),(177,'Brown',18,NULL,97,55.75,11.94,25.74),(178,'Brown',18,NULL,97,55.75,11.94,25.74),(179,'Brown',18,NULL,97,55.75,11.94,25.74),(180,'Deep',18,NULL,100,35.42,9.85,-55.96),(181,'Deep',18,NULL,100,35.42,9.85,-55.96),(182,'Brown',18,NULL,97,55.75,11.94,25.74),(183,'Brown',18,NULL,97,55.75,11.94,25.74),(184,'Deep',18,NULL,100,35.42,9.85,-55.96),(185,'Brown',18,NULL,97,55.75,11.94,25.74),(186,'Fair',18,NULL,99,63.78,17.03,38.98),(187,'Brown',18,NULL,97,55.75,11.94,25.74),(188,'Brown',18,NULL,97,55.75,11.94,25.74),(189,'Brown',18,NULL,97,55.75,11.94,25.74),(190,'Fair',18,NULL,100,82.42,16.08,63.62),(191,'Fair',18,NULL,100,82.42,16.08,63.62),(192,'Fair',18,NULL,100,82.42,16.08,63.62),(193,'Brown',18,NULL,97,55.75,11.94,25.74),(194,'Brown',18,NULL,97,55.75,11.94,25.74),(195,'Brown',18,NULL,97,55.75,11.94,25.74),(196,'Deep',18,NULL,100,35.42,9.85,-55.96),(197,'Brown',18,NULL,97,55.75,11.94,25.74),(198,'Brown',18,NULL,97,55.75,11.94,25.74),(199,'Deep',18,NULL,100,35.42,9.85,-55.96),(200,'Brown',18,NULL,97,55.75,11.94,25.74),(201,'Brown',18,NULL,97,55.75,11.94,25.74),(202,'Brown',18,NULL,97,55.75,11.94,25.74),(203,'Fair',18,NULL,100,82.42,16.08,63.62),(204,'Brown',18,NULL,97,55.75,11.94,25.74),(205,'Brown',18,NULL,97,55.75,11.94,25.74),(206,'Brown',18,NULL,97,55.75,11.94,25.74),(207,'Brown',18,NULL,97,55.75,11.94,25.74),(208,'Brown',18,NULL,97,55.75,11.94,25.74),(209,'Brown',18,NULL,97,55.75,11.94,25.74),(210,'Brown',18,NULL,97,55.75,11.94,25.74),(211,'Brown',18,NULL,97,55.75,11.94,25.74),(212,'Brown',18,NULL,97,55.75,11.94,25.74),(213,'Brown',18,NULL,97,55.75,11.94,25.74),(214,'Fair',18,NULL,100,82.42,16.08,63.62),(215,'Brown',18,NULL,97,55.75,11.94,25.74);
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

-- Dump completed on 2025-10-15 23:05:32
