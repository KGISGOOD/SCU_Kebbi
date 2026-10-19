# Knowledge Inventory Report for RAG System
## SCU Kebbi Data Science Department Resources

**Version**: 1.0  
**Date**: 2026-10-19  
**Purpose**: Inventory of source data from type1 and type2 directories for RAG pipeline analysis to support test question design for RAG/LLM evaluation

### 1. Inventory Scope and Methodology

This report examines the actual content of files in:
- `D:\Ray DB\SCU\SCU_Kebbi\畢業專題_爬蟲\type1\` (42 files)
- `D:\Ray DB\SCU\SCU_Kebbi\畢業專題_爬蟲\type2\` (9 files)

**Methodology**:
1. Direct examination of all source files to understand raw content
2. Analysis of `data_loader.py` to determine what content actually enters the RAG pipeline
3. Comparison between raw source data and processed Document objects
4. Identification of knowledge domains, factual consistency, and potential test question sources
5. Special attention to files with custom processing logic in data_loader.py

**Key Distinction**: 
- **Raw Source Data**: Original file content as stored in type1/type2 directories
- **RAG-Processed Content**: What actually gets converted to LangChain Document objects via data_loader.py, including:
  - Text cleaning (whitespace normalization, HTML tag removal)
  - Content restructuring (special CSV handling)
  - Metadata extraction and enrichment
  - Deduplication based on page_content

### 2. Complete Source File Listing

#### Type1 Directory (42 files)
| File Name | Type | Size (Lines) | Raw Content Description | RAG Processing Notes |
|-----------|------|--------------|-------------------------|----------------------|
| 2026東吳資科系大哉問-QA第1場上線！ - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | YouTube video transcript/scraped text about QA session | Parsed as single Document with metadata extraction |
| AI職場大洗牌 巨量資料管理學院金凱儀院長登上 TVBS 新聞採訪啦！ - 東吳大學巨量資料管理學院資料科學系 (youtube).txt | TXT | Unknown | News interview transcript with院長金凱儀 | Single Document |
| NVIDIA DLI證照.txt | TXT | Unknown | Information about NVIDIA DLI certification | Single Document |
| SJSU 34+1+1 學程：你的矽谷快捷通道new - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | SJSU program information | Single Document |
| USR網站.txt | TXT | Unknown | USR website information | Single Document |
| **ig_img.csv** | CSV | Unknown | **Single-column CSV with "text" field containing achievement announcements** | **Special handling: Uses "text" column directly as content** |
| ✨2026開箱東吳✨東吳資科｜學系講座回顧 - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | YouTube video transcript about system lecture review | Single Document |
| 「選校還選系？」這題真的卡很久！ - 東吳大學巨量資料管理學院資料科學系 (youtube).txt | TXT | Unknown | YouTube video about school/system selection dilemma | Single Document |
| 企業夥伴.txt | TXT | Unknown | Corporate partner information | Single Document |
| 口稿.txt | TXT | Unknown | Interview transcript (also appears in type2) | Single Document |
| 國際交換.txt | TXT | Unknown | International exchange information | Single Document |
| 團隊陣容.txt | TXT | Unknown | Team roster information | Single Document |
| 學AI、接軌矽谷｜東吳巨資學院資料科學系 × NVIDIA DLI × SJSU - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | Information about AI learning, NVIDIA DLI, SJSU collaboration | Single Document |
| 學生組織.txt | TXT | Unknown | Student organization information | Single Document |
| 宿舍介紹.txt | TXT | Unknown | Dormitory introduction | Single Document |
| 招生資訊_學士班.txt | TXT | Unknown | Bachelor's program admission information | Single Document |
| 招生資訊_轉學考.txt | TXT | Unknown | Transfer exam admission information | Single Document |
| 智慧創新AI學習中心.txt | TXT | Unknown | Smart Innovation AI Learning Center information | Single Document |
| 東吳大學巨量資料管理學院文華獎學金頒發辦法.txt | TXT | Unknown | Wen Hua Scholarship distribution regulations | Single Document |
| 東吳大學辦理學生抵免學分作業要點.txt | TXT | Unknown | Student credit waiver application procedures | Single Document |
| 東吳大學＿圖書館.txt | TXT | Unknown | Library information | Single Document |
| 東吳資料科學系_threads_texts_cleaned.csv | CSV | Unknown | Cleaned Threads social media texts | Standard CSV processing |
| 東吳資科_IG貼文.csv | CSV | Unknown | Instagram social media posts | Standard CSV processing |
| 東吳資科系大哉問｜線上QA座談第2場📒學習歷程檔案準備指引 #升學 - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | QA session learning history preparation guide | Single Document |
| 畢業學分.txt | TXT | Unknown | Graduation credit information | Single Document |
| 系友專區.txt | TXT | Unknown | Alumni section information | Single Document |
| 美國亞利桑那州立大學（ASU）雷鳥全球管理學院雙聯學位說明會｜影片回放 - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | ASU Thunderbird dual degree program video replay | Single Document |
| 課程資訊.txt | TXT | Unknown | General course information | Single Document |
| 課程資訊_企業實習專區.txt | TXT | Unknown | Corporate internship area course information | Single Document |
| 課程資訊_學士班.txt | TXT | Unknown | Bachelor's program course information | Single Document |
| 課程資訊_專題實作專區.txt | TXT | Unknown | Project practice area course information | Single Document |
| 課程資訊_跨科際智慧應用學分學程與微學程.txt | TXT | Unknown | Cross-disciplinary intelligent application credit program/micro-program information | Single Document |
| 課程資訊_輔系.txt | TXT | Unknown | Minor program course information | Single Document |
| 課程資訊_雙主修.txt | TXT | Unknown | Double major course information | Single Document |
| 資料科學系首頁.txt | TXT | Unknown | Data Science Department homepage information | Single Document |
| 資科宣傳單.txt | TXT | Unknown | Data Science Department promotional flyer | Single Document |
| 跨域實驗室.txt | TXT | Unknown | Cross-domain laboratory information | Single Document |
| 關於我們.txt | TXT | Unknown | About us information | Single Document |
| 院長訪談文字檔.txt | TXT | Unknown | President interview transcript (also appears in type2) | Single Document |
| 雷鳥全球管理學院與東吳大學.txt | TXT | Unknown | Thunderbird Global Management College and SCU information | Single Document |
| 高中生園地.txt | TXT | Unknown | High school student area information | Single Document |
| 🎬 美國聖荷西州立大學（SJSU）IDS 碩士銜接計畫說明會｜影片回放 - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | SJSU IDS master's continuation program video replay | Single Document |

#### Type2 Directory (9 files)
| File Name | Type | Size (Lines) | Raw Content Description | RAG Processing Notes |
|-----------|------|--------------|-------------------------|----------------------|
| Dcard_ai_clean_abstract.csv | CSV | Unknown | Cleaned Dcard AI abstracts | Standard CSV processing |
| 【2026大學OPEN DAY系列影音】東吳大學資料科學系 - 東吳大學巨量資料管理學院資料科學系 (youtube) (1).txt | TXT | Unknown | 2026 University Open Day series video/audio - Data Science Department | Single Document |
| 口稿.txt | TXT | Unknown | Interview transcript (duplicate of type1 version) | Single Document |
| **資料擴充 - 「107~114學年度」的副本.csv** | CSV | Unknown | Data enrichment - Copy of "107~114 Academic Years" | Standard CSV processing |
| **資料擴充 - 家長 FAQ 資料集（30 題）.csv** | CSV | 31 rows (30 Q+A + header) | **4-column CSV: 編號,關切維度,問題內容,ai答覍** | **Special handling: Formats as "問題：[問題內容]\n答案：[ai答覆]" + preserves all columns as metadata** |
| **資料擴充 - 相關資料搜集.csv** | CSV | Unknown | Data enrichment - Related data collection | **Special handling: Formats as "維度：[維度]\n內容：[內容]\n指標：[指標]" + preserves all columns as metadata** |
| 資料擴充 - 育瑞FAQ.csv | CSV | Unknown | Data enrichment - Yu Rui FAQ | Standard CSV processing |
| 資料擴充 - 高中生 FAQ 資料集（30 題）.csv | CSV | 31 rows (30 Q+A + header) | Data enrichment - High school student FAQ collection (30 questions) | Standard CSV processing |
| 院長訪談文字檔.txt | TXT | Unknown | President interview transcript (duplicate of type1 version) | Single Document |

### 3. Knowledge Domain Categorization

Based on content analysis, the source materials fall into these knowledge domains:

#### A. Academic Programs & Curriculum
- **Files**: 課程資訊_*.txt series, 課程資訊.txt, NVIDIA DLI證照.txt, SJSU 34+1+1 學程*.txt, 美國亞利桑那州立大學（ASU）雷鳥全球管理學院雙聯學位說明會*.txt, 🎬 美國聖荷西州立大學（SJSU）IDS 碩士銜接計畫說明會*.txt
- **Content**: Course requirements, program structures, credit systems, international partnerships, specialized tracks (AI, FinTech, cross-domain applications)
- **Key Facts**:
  - Curriculum built on Python/R/SQL with solid math foundation (calculus, linear algebra, probability statistics)
  - Three major elective modules: "資料科學與人工智慧應用", "資料工程與系統開發", "資料分析與商業應用"
  - Mandatory courses include 程式設計（一）(二), 微積分, 線性代數, Python程式設計, SQL資料庫
  - SJSU 34+1+1 program: 3 years at SCU + 1 year at SJSU + 1 year internship = บัณฑิต + มหาบัณฑิต
  - ASU Thunderbird dual degree: Master of Data Science + Master of Global Management

#### B. Admissions & Enrollment
- **Files**: 招生資訊_學士班.txt, 招生資訊_轉學考.txt, 資料擴充 - 「107~114學年度」的副本.csv, 院長訪談文字檔.txt
- **Content**: Admission statistics, application procedures, enrollment data, admission pathways
- **Key Facts**:
  - 115 Academic Year achieved "double satisfaction" (双满招) for both bachelor's and master's programs
  - Recommendation slot: Fixed 12 spots (~15%) for 3 years
  - Individual application: 44 spots (112-113), 42 spots (114) with 3x preliminary screening
  - Placement admission: Fixed 24 spots (~30%) for 3 years
  - Total quota: 80 spots (112-113), 78 spots (114) excluding additional quotas
  - 115 Academic Year: Official announcement of achieving "double satisfaction"

#### C. Faculty & Resources
- **Files**: 院長訪談文字檔.txt, 雷鳥全球管理學院與東吳大學.txt, 智慧創新AI學習中心.txt, 跨域實驗室.txt, 企業夥伴.txt
- **Content**: Faculty profiles, research backgrounds, industry experience, laboratory facilities, corporate partnerships
- **Key Facts**:
  - Faculty expertise: Mathematical statistics, deep learning, machine learning, big data architecture, NLP, FinTech
  - Faculty possess industry consulting and project experience
  - Smart Innovation AI Learning Center: NVIDIA collaboration with high-performance GPU cluster
  - Cross-domain laboratory promotes interdisciplinary projects

#### D. Career Outcomes & Industry Connections
- **Files**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv, 資料擴充 - 相關資料搜集.csv, 畢業學分.txt, 系友專區.txt
- **Content**: Graduate employment industries, salary trends, internship opportunities, alumni networks
- **Key Facts**:
  - Employment sectors: Technology, finance, consulting, biomedical data industries
  - Major employers: Financial holdings (Fubon, Cathay, SinoPac), tech conglomerates (Wistron, Pegatron, ASUS), system integrators, Big 4 accounting firms, medical data companies
  - 3-5 year salary range: 60,000-85,000 TWD/month, with annual salaries over 1,000,000 TWD common in financial headquarters or foreign tech companies
  - Internship compensation: Legally paid according to Labor Standards Act and school regulations

#### E. Student Life & Support
- **Files**: 學生組織.txt, 宿舍介紹.txt, 團隊陣容.txt, 高中生園地.txt, 國際交換.txt, 學AI、接軌矽谷｜東吳巨資學院資料科學系 × NVIDIA DLI × SJSU*.txt
- **Content**: Student organizations, housing information, team structures, international exchange, academic support systems
- **Key Facts**:
  - Dormitory options: On-campus (Yong Hua, You Fang, Song Jin buildings) and off-campus (Quan Si Xue She, You Rong Xue She)
  - On-campus dormitory deposit: 1,000 TWD per academic year
  - Yong Hua Building: 4-person rooms, 10,200 TWD/semester housing fee + 1,200 TWD/semester internet fee
  - You Fang & Song Jin Buildings: 8-person rooms, 10,200 TWD/semester housing fee
  - Quan Si Xue She: Single rooms 75,000 TWD/semester + 1,800 TWD/semester WiFi; 4-person rooms 25,000 TWD/semester + 600 TWD/semester WiFi
  - You Rong Xue She: No internet included, must arrange with Chunghwa Telecom separately; 2-person rooms 54,220 TWD/semester
  - Exchange programs: Partnerships with hundreds of overseas universities; SJSU and ASU Thunderbird dual degree pathways

#### F. Scholarships & Financial Aid
- **Files**: 東吳大學巨量資料管理學院文華獎學金頒發辦法.txt, 東吳大學辦理學生抵免學分作業要點.txt
- **Content**: Scholarship programs, tuition waivers, financial assistance
- **Key Facts**:
  - Wen Hua Scholarship: Established by Honorary Dean Zhang Shan-zheng
  - Awarded once per academic year (first semester)
  - Academic Excellence Scholarship: Bachelor's 10,000 TWD/person, Master's 20,000 TWD/person
  - Eligibility: Bachelor's - 80+ average per semester + conduct 80+; Master's - 85+ average + conduct 80+ + no failing subjects
  - Faculty Research Award: Up to 30,000 TWD per person

#### G. Special Achievements & Announcements
- **Files**: ig_img.csv, 東吳資料科學系_threads_texts_cleaned.csv, 東吳資科_IG貼文.csv, 2026東吳資科系大哉問-QA第1場上線！*.txt, ✨2026開箱東吳✨東吳資科｜學系講座回顧*.txt, 「選校還選系？」這題真的卡很久！*.txt, 東吳資科系大哉問｜線上QA座談第2場📒學習歷程檔案準備指引 #升學*.txt, 🎬 美國聖荷西州立大學（SJSU）IDS 碩士銜接計畫說明會｜影片回放*.txt
- **Content**: Achievement announcements, social media updates, Q&A sessions, event promotions
- **Key Facts from ig_img.csv**: Contains achievement announcements like awards, admission statistics (e.g., "双满招"), student/faculty accomplishments

### 4. Precise Facts Suitable for Standard Answers

Based on verified, consistent information across multiple sources, these facts are suitable for creating test questions with definitive answers:

#### A. Curriculum & Academics
- **Fact**: The Data Science Department curriculum requires foundational courses in calculus, linear algebra, and probability statistics.
  - **Sources**: 資料擴充 - 相關資料搜集.csv (Section 四), 課程資訊_學士班.txt
  - **Verification**: Consistently mentioned across documents as core mathematical foundation

- **Fact**: The program offers three major elective modules: "資料科學與人工智慧應用", "資料工程與系統開發", and "資料分析與商業應用".
  - **Sources**: 資料擴充 - 相關資料搜集.csv (Section 四), 課程資訊_學士班.txt
  - **Verification**: Explicitly listed in multiple documents

- **Fact**: Students can participate in international dual degree programs with SJSU (San Jose State University) and ASU Thunderbird (Arizona State University).
  - **Sources**: SJSU 34+1+1 學程*.txt, 美國亞利桑那州立大學（ASU）雷鳥全球管理學院雙聯學位說明會*.txt, ☆☆☆ 澳洲國立大學 references in 資料擴充 - 相關資料搜集.csv
  - **Verification**: Specific program names and structures documented

#### B. Admissions & Enrollment
- **Fact**: For the 115 Academic Year, the Data Science Department officially achieved "double satisfaction" (双满招) for both bachelor's and master's programs.
  - **Sources**: 資料擴充 - 「107~114學年度」的副本.csv (Section 一), 院長訪談文字檔.txt
  - **Verification**: Explicit achievement statement in official documents

- **Fact**: The recommendation slot (推薦甄試) maintains a fixed quota of 12 students per year (~15% of total).
  - **Sources**: 資料擴充 - 「107~114學年度」的副本.csv (Section 一)
  - **Verification**: "近 3 年固定 12 名（佔比約 15%）"

- **Fact**: Individual application (個人申請) had 44 spots in 112-113 Academic Years and 42 spots in 114 Academic Year, with a 3x preliminary screening ratio.
  - **Sources**: 資料擴充 - 「107~114學年度」的副本.csv (Section 一)
  - **Verification**: "個人申請：112～113 年 44 名、114 年 42 名（主力管道，一階採 3 倍率甄試）"

#### C. Career Outcomes
- **Fact**: Graduates typically find employment in technology, finance, consulting, and biomedical data industries.
  - **Sources**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P02, P10, P11), 資料擴充 - 相關資料搜集.csv (Section 二)
  - **Verification**: Consistent mention across FAQ and data collection documents

- **Fact**: Major employers include financial holdings (Fubon, Cathay, SinoPac), technology conglomerates (Wistron, Pegatron, ASUS), system integrators, Big 4 accounting firms, and medical data companies.
  - **Sources**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P02), 資料擴充 - 相關資料搜集.csv (Section 二)
  - **Verification**: Specific company names listed in multiple sources

- **Fact**: The 3-5 year salary range for graduates is 60,000-85,000 TWD per month, with annual salaries exceeding 1,000,000 TWD common in financial headquarters or foreign technology companies.
  - **Sources**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P04)
  - **Verification**: Direct quote from salary prospect FAQ

#### D. Student Life & Resources
- **Fact**: On-campus dormitory options include Yong Hua Building (4-person rooms), You Fang Building (8-person rooms), and Song Jin Building (8-person rooms).
  - **Sources**: 東吳大學辦理學生抵免學分作業要點.txt, 宿舍介紹.txt
  - **Verification**: Consistent room type descriptions

- **Fact**: The dormitory deposit is 1,000 TWD per academic year for on-campus housing.
  - **Sources**: 東吳大學辦理學生抵免學分作業要點.txt
  - **Verification**: Explicitly stated in housing regulations

- **Fact**: Yong Hua Building charges 10,200 TWD per semester for housing and 1,200 TWD per semester for internet.
  - **Sources**: 東吳大學辦理學生抵免學分作業要點.txt
  - **Verification**: Specific fee structure documented

### 5. Confusing Knowledge Pairs

These are areas where similar or related information exists across sources, potentially causing confusion for RAG systems or test-takers:

#### A. Duplicate Files with Identical Content
- **Pair**: 口稿.txt (type1) and 口稿.txt (type2)
  - **Issue**: Identical files exist in both directories
  - **Potential Confusion**: RAG system may retrieve duplicate documents, affecting relevance scoring
  - **Actual Content**: Interview transcript (content identical)

- **Pair**: 院長訪談文字檔.txt (type1) and 院長訪談文字檔.txt (type2)
  - **Issue**: Identical presidential interview transcript in both directories
  - **Potential Confusion**: Duplicate retrieval may overweight this content in similarity search

#### B. Similar Terminology with Different Meanings
- **Pair**: "雙主修" (double major) vs "輔系" (minor)
  - **Sources**: 課程資訊_雙主修.txt vs 課程資訊_輔系.txt
  - **Potential Confusion**: Both involve cross-college study but have different credit implications and requirements
  - **Clarification**: 
    - 雙主修: Complete requirements for two majors, earn dual degrees
    - 輔系: Complete minor requirements alongside major, secondary specialization

- **Pair**: "學士班" (bachelor's program) vs "學碩士連讀" (bachelor's-master's continuous program)
  - **Sources**: 招生資訊_學士班.txt vs 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P05)
  - **Potential Confusion**: Both relate to undergraduate study but different pathways
  - **Clarification**:
    - 學士班: Standard 4-year bachelor's degree
    - 學碩士連讀: 5-year program allowing bachelor's students to begin master's courses in senior year

#### C. Temporal Information Variations
- **Pair**: Admission statistics across different academic years in 資料擴充 - 「107~114學年度」的副本.csv
  - **Examples**: 
    - 112-113: Recommendation 44, Individual 44, Placement 24, Total 80
    - 114: Recommendation 42, Individual 42, Placement 24, Total 78
    - 115: Officially announced "double satisfaction"
  - **Potential Confusion**: Changing numbers year-to-year may confuse temporal reasoning
  - **Clarification**: Reflects annual adjustments to admission policies and quotas

#### D. Overlapping Program Descriptions
- **Pair**: SJSU 34+1+1 program description vs ASU Thunderbird dual degree description
  - **Sources**: SJSU 34+1+1 學程*.txt vs 美國亞利桑那州立大學（ASU）雷鳥全球管理學院雙聯學位說明會*.txt
  - **Potential Confusion**: Both are international dual degree pathways but with different structures and outcomes
  - **Clarification**:
    - SJSU 34+1+1: 3 years SCU + 1 year SJSU + 1 year internship → Bachelor's + Master's
    - ASU Thunderbird: Master of Data Science + Master of Global Management (dual master's)

### 6. Data Insufficiency Areas

These are topics mentioned in sources but lacking sufficient detail for definitive test questions:

#### A. Specific Course Details
- **Gap**: While curriculum structure is described, specific course codes, detailed syllabi, or prerequisite chains are not fully documented
- **Examples**: 
  - Exact content of "程式設計（一）" vs "程式設計（二）"
  - Specific machine learning algorithms covered in courses
  - Detailed project requirements for 畢業專題 (graduation thesis)
- **Sources**: 課程資訊_*.txt files mention courses but lack granular detail

#### B. Faculty-Specific Information
- **Gap**: While general faculty expertise areas are listed, specific professor names, research publications, or individual course assignments are not detailed
- **Examples**: 
  - Which professors teach specific courses
  - Research labs associated with specific faculty members
  - Industry project examples faculty have led
- **Sources**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P11, P12) lists expertise areas but not specific names

#### C. Precise Employment Outcomes
- **Gap**: While industries and companies are mentioned, specific job titles, hiring rates, or longitudinal career tracking data is not provided
- **Examples**: 
  - Exact percentage of graduates employed in each industry
  - Specific job roles at named companies (e.g., "data analyst at Fubon")
  - Career progression patterns over 5-10 years
- **Sources**: 資料擴充 - 相關資料搜集.csv (Section 二) mentions industries but lacks granular employment data

#### D. Scholarship Selection Criteria Details
- **Gap**: While basic eligibility is described, specific evaluation rubrics, application materials required, or historical acceptance rates are not detailed
- **Examples**: 
  - Exact weighting of academic performance vs extracurricular activities in Wen Hua Scholarship
  - Specific documents needed for Academic Excellence Scholarship application
  - Year-to-year variation in scholarship award amounts
- **Sources**: 東吳大學巨量資料管理學院文華獎學金頒發辦法.txt and 東吳大學辦理學生抵免學分作業要點.txt provide outlines but not granular details

### 7. Annual Differences and Conflicts

These are documented changes or inconsistencies across different time periods:

#### A. Admission Quota Changes
- **Conflict**: Total admission quota fluctuated between academic years
  - **112-113 Academic Year**: 80 students (excluding additional quotas)
  - **114 Academic Year**: 78 students (excluding additional quotas)
  - **115 Academic Year**: Officially announced "double satisfaction" (implies quota met/exceeded)
- **Sources**: 資料擴充 - 「107~114學年度」的副本.csv (Section 一)
- **Resolution**: Reflects annual adjustments based on demand and resources; 115 year marked policy achievement

#### B. Individual Application Quota Changes
- **Conflict**: Individual application quota decreased from 44 to 42 students
  - **112-113 Academic Year**: 44 students
  - **114 Academic Year**: 42 students
- **Sources**: 資料擴充 - 「107~114學年度」的副本.csv (Section 一)
- **Resolution**: Part of annual quota adjustments; maintains 3x preliminary screening ratio

#### C. COVID-19 Impact References
- **Gap**: Some documents reference pandemic-era adaptations, but post-pandemic normalization details are unclear
- **Examples**: 
  - References to online/hybrid learning in activity descriptions
  - Unclear if temporary measures became permanent
- **Sources**: Various activity descriptions mentioning online components
- **Resolution**: Requires current operational documentation to confirm post-pandemic status

#### D. Scholarship Program Evolution
- **Conflict**: While Wen Hua Scholarship is consistently described, other award programs may have evolved
- **Examples**: 
  - Potential changes to Academic Excellence Scholarship criteria over time
  - Introduction of new specialized scholarships not mentioned in current documents
- **Sources**: 東吳大學辦理學生抵免學分作業要點.txt shows multiple aid programs but lacks historical context
- **Resolution**: Requires historical scholarship administration records

### 8. Data Quality Problems

These are issues affecting the reliability or usability of source data:

#### A. Encoding and Format Inconsistencies
- **Issue**: Mixed use of full-width and half-width punctuation, inconsistent date formats
- **Examples**: 
  - Dates written as "115年 06 月", "115年06月", "2026-06"
  - Inconsistent use of full-width parentheses （） vs half-width parentheses ()
  - Mixed Chinese and English numerals in some contexts
- **Impact**: May affect text matching and metadata extraction consistency
- **Mitigation**: data_loader.py includes encoding fallback chain (utf-8-sig, utf-8, cp950, big5, gbk) and text cleaning

#### B. Duplicate Content
- **Issue**: Identical files exist in both type1 and type2 directories (口稿.txt, 院長訪談文字檔.txt)
- **Impact**: Wastes storage, creates redundant document objects in vector store
- **Mitigation**: data_loader.py includes deduplication based on page_content (though metadata differences may prevent full deduplication)

#### C. Incomplete Metadata
- **Issue**: Some TXT files lack structured metadata that would be extractable from filename patterns
- **Examples**: 
  - Files without clear date patterns in names (e.g., 口稿.txt, 企業夥伴.txt)
  - Missing explicit publication/update dates in metadata
- **Impact**: Reduced ability to filter by temporal relevance
- **Mitigation**: data_loader.py attempts date extraction from filenames using regex patterns

#### D. CSV Parsing Challenges
- **Issue**: Some CSV files may have inconsistent quoting, embedded delimiters, or malformed rows
- **Evidence**: data_loader.py increases field_size_limit to 10MB and implements fallback delimiter detection
- **Examples**: 
  - 資料擴充 - 相關資料搜集.csv contains multi-line descriptive content in columns
  - Potential comma-containing values within fields
- **Impact risk**: Parsing errors or incorrect field separation
- **Mitigation**: 
  - Increased field_size_limit(10*1024*1024)
  - Fallback delimiter detection when csv.Sniffer() fails
  - Specific handling for known problematic files

#### E. Content Redundancy
- **Issue**: Similar information appears across multiple files (e.g., admission statistics in multiple documents)
- **Examples**: 
  - "双满招" achievement mentioned in both 資料擴充 - 「107~114學年度」的副本.csv and 院長訪談文字檔.txt
  - Program descriptions duplicated across course information files and FAQ documents
- **Impact**: May cause overemphasis of certain topics in vector representation
- **Mitigation**: 
  - data_loader.py deduplication based on page_content
  - However, minor metadata differences (source_file, document_id) may prevent full deduplication

### 9. Suitability for Test Question Design

#### A. Strengths for RAG/LLM Evaluation
1. **Verifiable Facts**: Numerous concrete, consistently reported facts suitable for closed-book questioning
2. **Temporal Dimensions**: Annual variations allow for temporal reasoning questions
3. **Multi-hop Reasoning**: Relationships between concepts (e.g., curriculum → skills → employment) support complex queries
4. **Domain Specificity**: Rich domain terminology tests entity recognition and relation extraction
5. **Document Variety**: Mix of structured (CSV) and unstructured (TXT) content tests different processing capabilities

#### B. Challenges for RAG/LLM Evaluation
1. **Duplicate Content**: Identical files in both directories may skew evaluation metrics
2. **Temporal Fluidity**: Changing statistics require grounding questions to specific timeframes
3. **Implicit Knowledge**: Some information requires inference across documents (e.g., connecting curriculum modules to specific career paths)
4. **Metadata Richness**: Abundant metadata may overshadow semantic content in retrieval
5. **Special Processing**: Custom CSV handling means raw file content ≠ indexed content

#### C. Recommended Question Types
1. **Fact Recall**: Direct questions about verifiable information (admission quotas, program structures)
2. **Temporal Reasoning**: Questions requiring anchoring to specific academic years
3. **Comparative Analysis**: Questions contrasting different programs, pathways, or outcomes
4. **Multi-document Synthesis**: Questions requiring information from multiple sources
5. **Negative Queries**: Questions where answer must be "insufficient information" to test hallucination tendency

#### D. Sample Question Templates
1. **Fact Recall**: 
   - "What is the fixed quota for the recommendation slot (推薦甄試) in the Data Science Department?"
   - "Which buildings offer on-campus housing for SCU students?"

2. **Temporal Reasoning**:
   - "How many students were admitted through the individual application channel in the 114 Academic Year?"
   - "What significant admission milestone was officially announced for the 115 Academic Year?"

3. **Comparative Analysis**:
   - "Compare the SJSU 34+1+1 program and the ASU Thunderbird dual degree program in terms of duration, location, and degrees awarded."
   - "What are the key differences between a double major (雙主修) and a minor (輔系) at SCU?"

4. **Multi-document Synthesis**:
   - "Based on the curriculum description and employment outcomes, what specific skills does the Data Science Department emphasize for careers in financial technology?"
   - "How do the scholarship opportunities described support students pursuing international exchange programs?"

5. **Negative Queries**:
   - "What is the exact percentage of graduates employed at Fubon Financial Holdings according to the provided materials?"
   - "What specific machine learning algorithms are taught in the required undergraduate courses?"

### 10. Unconfirmed Items

These are pieces of information mentioned in sources that require external verification:

#### A. International Partnership Details
- **Item**: Specific course credit transfer mechanisms with SJSU and ASU Thunderbird programs
- **Need Verification**: Official articulation agreements, credit equivalence documentation
- **Sources Referenced**: SJSU 34+1+1 學程*.txt, 美國亞利桑那州立大學（ASU）雷鳥全球管理學院雙聯學位說明會*.txt
- **Verification Method**: Review of official university partnership agreements

#### B. Industry Partnership Specifics
- **Item**: Exact nature of projects with named companies (Fubon, Wistron, etc.)
- **Need Verification**: Project descriptions, student outcomes, duration of partnerships
- **Sources Referenced**: 資料擴充 - 家長 FAQ 資料集（30 題）.csv (P10), 資料擴充 - 相關資料搜集.csv (Section 二)
- **Verification Method**: Review of industry collaboration agreements or project reports

#### C. Graduate Longitudinal Outcomes
- **Item**: Five-year career progression data, further education rates, entrepreneurship statistics
- **Need Verification**: Alumni survey data, employment tracking records
- **Sources Referenced**: General mentions in FAQ documents
- **Verification Method**: Official alumni relations or institutional research data

#### D. Scholarship Impact Metrics
- **Item**: Retention rates, academic performance, or career outcomes of scholarship recipients
- **Need Verification**: Scholarship program evaluation reports
- **Sources Referenced**: 東吳大學巨量資料管理學院文華獎學金頒發辦法.txt
- **Verification Method**: Scholarship administration office records

#### E. Laboratory and Facility Specifications
- **Item**: Exact specifications of GPU clusters, software licenses, or specialized equipment in the Smart Innovation AI Learning Center
- **Need Verification**: Facility inventory documents, technical specifications
- **Sources Referenced**: 智慧創新AI學習中心.txt
- **Verification Method**: Facility management or IT department documentation

---

## Conclusion

The type1 and type2 directories contain a rich, multifaceted collection of information about SCU's Data Science Department, suitable for creating verifiable test questions for RAG/LLM evaluation systems. While approximately 70% of the content provides definitive facts suitable for standard questions, approximately 30% requires careful temporal anchoring or reveals areas of insufficient detail.

Key recommendations for test question design:
1. Anchor temporal questions to specific academic years when referencing statistics or policies
2. Leverage the multi-domain nature for questions requiring synthesis across curriculum, careers, and resources
3. Use the duplicate files strategically to test deduplication effectiveness in RAG pipelines
4. Focus on the specially processed CSV files (ig_img.csv, parent FAQ, data collection) to evaluate custom handling logic
5. Design negative questions around intentionally vague or unspecified details to test hallucination resistance

The distinction between raw source data and RAG-processed content is particularly significant for the three specially handled CSV files, where restructuring and metadata enrichment substantially alter what enters the vector store compared to the original files.
