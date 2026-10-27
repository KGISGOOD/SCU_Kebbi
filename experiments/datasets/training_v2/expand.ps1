param([string]$ProjectRoot=(Join-Path $PSScriptRoot '../../..'))
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($ProjectRoot)
$dir=Join-Path $root 'experiments/datasets/training_v2'
$utf=[Text.UTF8Encoding]::new($false,$true)
function ReadText($p){$utf.GetString([IO.File]::ReadAllBytes($p)).TrimStart([char]0xFEFF)}
function Hash($p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function WriteJson($p,$o){[IO.File]::WriteAllText($p,($o|ConvertTo-Json -Depth 40)+[char]10,$utf)}
function WriteRows($p,$o){[IO.File]::WriteAllText($p,((@($o)|ForEach-Object{$_|ConvertTo-Json -Compress -Depth 40})-join [char]10)+[char]10,$utf)}
$manifest=ReadText (Join-Path $dir 'source_manifest.json')|ConvertFrom-Json
$texts=@{};$sm=@{}
foreach($s in $manifest.sources){
    if($s.path -notmatch '^畢業專題_爬蟲/type[12]/'){throw 'Source outside allowed roots'}
    $full=[IO.Path]::GetFullPath((Join-Path $root $s.path))
    if(!$full.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Source escapes workspace'}
    if((Hash $full) -cne $s.sha256){throw "Changed source: $($s.path)"}
    $texts[$s.path]=ReadText $full;$sm[$s.path]=$s
}
function Locate($path,$anchor,$end){
    $t=$texts[$path];if($null -eq $t){throw "Unregistered source: $path"}
    $a=$anchor;$start=$t.IndexOf($a,[StringComparison]::Ordinal)
    if($start -lt 0){$a=$a.Replace("`r`n","`n");$start=$t.IndexOf($a,[StringComparison]::Ordinal)}
    if($start -lt 0){throw "Missing start anchor $path : $anchor"}
    $stop=$t.Length
    if($end){$e=$end;$stop=$t.IndexOf($e,$start+$a.Length,[StringComparison]::Ordinal);if($stop -lt 0){$e=$e.Replace("`r`n","`n");$stop=$t.IndexOf($e,$start+$a.Length,[StringComparison]::Ordinal)};if($stop -lt 0){throw "Missing end anchor $path : $end"}}
    [pscustomobject][ordered]@{path=$path;sha256=$sm[$path].sha256;start_utf16=$start;length_utf16=$stop-$start;line_start=1+[regex]::Matches($t.Substring(0,$start),'\r\n|\r|\n|\u2028').Count;line_end=1+[regex]::Matches($t.Substring(0,$stop),'\r\n|\r|\n|\u2028').Count;quote=$t.Substring($start,$stop-$start);offset_basis='decoded UTF-8 without BOM; UTF-16 code units; original newlines retained'}
}
$current=@((ReadText (Join-Path $dir 'qa_master.jsonl')) -split '\r?\n'|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json})
# Retain and revalidate the original facts, not copies of a previous generated workbook.
$rows=[Collections.Generic.List[object]]::new()
foreach($r in $current){if($r.intent_group_id -match '^K(\d+)$' -and [int]$Matches[1] -le 32 -or $r.intent_group_id -match '^M00[1-4]$'){$rows.Add($r)}}
if($rows.Count -ne 54){throw 'Expected the 54 first-edition candidates; do not reconstruct missing evidence'}
$spec=ReadText (Join-Path $dir 'expansion_spec.json')|ConvertFrom-Json
foreach($p in $spec.points){
    $path='畢業專題_爬蟲/type1/'+$p.file
    $e=@(Locate $path $p.anchor $p.end)
    foreach($other in $p.additional_evidence){$e+=Locate ('畢業專題_爬蟲/type1/'+$other.file) $other.anchor $other.end}
    $year=$null
    if($p.topic -eq '112入學課程'){$year=112;$e+=Locate $path '（適用112學年度一年級入學新生）' '【畢業學分規定】'}
    if($e[0].quote -notlike ('*'+$p.claim+'*')){throw "Claim not supported at selected location: $($p.id) $($p.claim)"}
    if(!$p.answer.Contains([string]$p.claim)){throw "Claim absent from answer: $($p.id)"}
    $ids=@();for($i=1;$i -le $p.questions.Count;$i++){$ids+='TV2_'+$p.id+'_'+$i.ToString('00')}
    $pairs=@();for($i=1;$i -lt $ids.Count;$i++){$pairs+=[pscustomobject]@{pair_id=('PP_'+$p.id+'_'+($i+1));query_id=$ids[$i];positive_id=$ids[0];intent_group_id=$p.id;semantic_status='assistant_clause_and_scope_reviewed_not_independently_certified'}}
    $family=switch($p.topic){'112入學課程'{'curriculum_112'} default {'topic_'+$p.topic}}
    for($i=0;$i -lt $ids.Count;$i++){
        $r=[pscustomobject][ordered]@{
            question_id=$ids[$i];question=$p.questions[$i];answer=$p.answer;qa_type=if($i){'paraphrase'}else{'basic_qa'};
            intent_group_id=$p.id;knowledge_point_ids=@($p.id);leakage_family_id=$family;
            intent_signature=[pscustomobject]@{subject=$p.questions[0];attribute=$p.id;scope=$p.scope};
            canonical_question_id=$ids[0];paraphrase_ids=@($ids|Where-Object{$_ -ne $ids[$i]});
            positive_pairs=@($pairs|Where-Object{$_.query_id -eq $ids[$i] -or $_.positive_id -eq $ids[$i]});hard_negative_pairs=@();
            pair_roles=@(if($pairs.Count){if($i){'positive_query'}else{'positive_document'}});sources=$e;
            answer_claims=@([pscustomobject]@{value=$p.claim;evidence_indexes=@(0);rule='literal_value_present_in_answer_and_quoted_source_not_full_entailment'});
            question_invariants=@(if($year){[string]$year};foreach($token in $p.required_question_tokens){if($token){$token}});applicable_period=$p.scope;required_conditions=@($p.scope);
            academic_year=$year;scope_inheritance='never_inherit_year_from_previous_question';
            temporal_status=if($year){'explicit_cohort_or_academic_year'}elseif($p.topic -eq '大學社會責任' -or $p.historical){'explicit_historical_fact'}else{'undated_snapshot_not_current_policy_certified'};
            training_uses=@(if($pairs.Count){'A2-1';'A2-2'};'A3');split=$null;review_status='not_human_approved';
            auto_validation=[pscustomobject]@{evidence_checks='pending';conditions_checks='pending';semantic_equivalence='unverified_by_independent_method';negative_semantics='unverified_by_independent_method';contradiction_checks='bounded_rules_only';formal_training_eligible=$false};
            dataset_status='source_grounded_candidate_not_formal_training_approval';required_answer_phrases=@($p.qualifiers);
            topic=$p.topic;
            semantic_source_review=[pscustomobject]@{method='assistant_reading_of_cited_rule_clauses_not_independent_validator';answer_support='all_answer_clauses_compared_to_cited_section';paraphrase_review='same_requested_information_and_answer_scope';preserved_scope=$p.scope;residual_risk='undated_rules_are_not_certified_current_policy; no human approval or independent entailment proof'}
        };$rows.Add($r)
    }
}
# Explicit contrasts are reviewed at the information-request level, not inferred from cosine scores.
$contrasts=@(
    @('K038','K039','同為課程學分互抵，但以多抵少與以少抵多的處理方向相反。'),
    @('K040','K041','同為抵免申請，一題問辦理時間及追抵限制，一題問成績證明與送件對象。'),
    @('K045','K046','同為文華學生獎學金，一題問學業獎固定金額，一題問競賽與研究獎金決定方式。'),
    @('K048','K049','文華競賽資格要求公開競賽成果；研究資格要求研討會或有審稿制度期刊，證明不能互換。'),
    @('K051','K052','同為文華申請附件，學業獎要求歷年成績單，競賽獎要求特定期間競賽證明及事由成果。'),
    @('K052','K053','同為文華申請附件，競賽成果與研究論文發表證明用途不同。'),
    @('K056','K057','同為跨領域學程申請，一題問學士班開始年級，一題問研究生及他校學生資格。'),
    @('K058','K061','同為學程學分，一題問最低修習量與外主系比例，一題問主系與學程互抵上限。'),
    @('K074','K075','同為專題時程，一題問申請與製作期，一題問成果發表時點。'),
    @('K078','K079','同為專題成果格式，一題問海報，一題問影片；規格不能互用。'),
    @('K084','K085','同為全學年課程，一題問下學期班別，一題問學分認列條件。'),
    @('K100','K101','同為學習方式，問題導向針對大一基礎培訓，專案導向透過業界專案解決產業問題。'),
    @('K103','K104','同為招生應用領域內容，智慧應用的資料工程／機器學習清單，不能替代服務科學的服務設計與行銷清單。'),
    @('K103','K105','同為招生應用領域內容，智慧應用的技術清單，不能替代金融科技的智能交易與風險管理清單。'),
    @('K104','K105','同為招生應用領域內容，服務科學的設計與行銷清單，不能替代金融科技的交易與風險管理清單。'),
    @('K114','K115','同為三大模組的技術能力，AI模型訓練整合與資料管線／後端／雲端架構的資訊需求不同；部分部署詞彙重疊不表示整份答案可互換。'),
    @('K114','K116','同為三大模組的技術能力，AI系統的模型開發與商務分析的金融建模／時間序列需求不同。'),
    @('K115','K116','同為三大模組的技術能力，資料工程的管線／雲端架構與商務的金融建模／決策分析需求不同。'),
    @('K121','K122','同為兼任教師授課科目，宋浩與劉豐泰是不同姓名，來源分別列機器學習及資料庫管理與實務，不能互換教師對應。'),
    @('K121','K123','同為兼任教師授課科目，宋浩與蕭宇程是不同姓名，來源分別列機器學習及區塊鏈，不能互換教師對應。'),
    @('K122','K123','同為兼任教師授課科目，劉豐泰與蕭宇程是不同姓名，來源分別列資料庫管理與實務及區塊鏈，不能互換教師對應。')
)
$by=@{};foreach($r in $rows){$by[$r.question_id]=$r}
foreach($c in $contrasts){
    foreach($direction in @(0,1)){
        $a=$by['TV2_'+$c[$direction]+'_02'];$b=$by['TV2_'+$c[1-$direction]+'_01']
        if(!$a -or !$b -or $a.leakage_family_id -cne $b.leakage_family_id){throw "Invalid contrast IDs/family: $($c -join '|')"}
        $pair=[pscustomobject]@{pair_id=('HN_'+$a.intent_group_id+'_'+$b.intent_group_id);anchor_id=$a.question_id;negative_id=$b.question_id;anchor_intent_id=$a.intent_group_id;negative_intent_id=$b.intent_group_id;rationale=$c[2];semantic_status='assistant_information_request_contrast_reviewed_not_independently_certified';model_difficulty='not_measured';split_policy='retain_only_if_both_intents_are_future_train'}
        $a.hard_negative_pairs=@($a.hard_negative_pairs)+$pair;$b.hard_negative_pairs=@($b.hard_negative_pairs)+$pair
    }
}
foreach($r in $rows){
    foreach($field in @('knowledge_point_ids','pair_roles','training_uses','required_conditions','question_invariants','required_answer_phrases','paraphrase_ids','positive_pairs','hard_negative_pairs','sources','answer_claims')){$r.$field=@($r.$field)}
    if($r.intent_group_id -match '^K(\d+)$' -and [int]$Matches[1] -le 32 -or $r.intent_group_id -match '^M00[1-4]$'){
        $topic=if($r.sources.path -contains '畢業專題_爬蟲/type1/畢業學分.txt'){'112入學課程'}elseif($r.sources.path -contains '畢業專題_爬蟲/type1/宿舍介紹.txt'){'115住宿'}elseif($r.sources.path -contains '畢業專題_爬蟲/type1/課程資訊_企業實習專區.txt'){'企業實習'}elseif($r.sources.path -contains '畢業專題_爬蟲/type1/課程資訊_專題實作專區.txt'){'專題歷年成果'}else{'實驗室與資源'}
        $r|Add-Member -Force topic $topic
    }
    if(!$r.PSObject.Properties['topic']){
        $topic=switch -Regex($r.leakage_family_id){'curriculum'{'112入學課程'} 'housing|dorm'{'115住宿'} 'intern'{'企業實習'} 'project'{'專題歷年成果'} 'lab|research|ai_learning'{'實驗室與資源'} default {'既有整合與AI資源'}}
        $r|Add-Member topic $topic
    }
    if(!$r.PSObject.Properties['semantic_source_review']){$r|Add-Member semantic_source_review ([pscustomobject]@{method='assistant_recheck_of_existing_question_answer_and_cited_evidence';answer_support='source_scoped_candidate';residual_risk='no independent semantic or current-version certification'})}
    # Keep overlapping facts from the same documents connected for a later group-wise split.
    if($r.sources.path -contains '畢業專題_爬蟲/type1/課程資訊_企業實習專區.txt'){$r.leakage_family_id='internship_documents'}
    if($r.sources.path -contains '畢業專題_爬蟲/type1/課程資訊_專題實作專區.txt'){$r.leakage_family_id='project_documents'}
    if($r.sources.path -contains '畢業專題_爬蟲/type1/跨域實驗室.txt'){$r.leakage_family_id='laboratories'}
    if($r.sources.path -contains '畢業專題_爬蟲/type1/智慧創新AI學習中心.txt'){$r.leakage_family_id='ai_learning'}
    $r|Add-Member -Force negative_exclusion_intent_ids @()
}
# Parent/child requests and overlapping descriptions are not silently treated as negatives.
$overlapExclusions=@(@('K102','K103'),@('K102','K104'),@('K102','K105'),@('K103','K114'),@('K103','K115'),@('K104','K116'),@('K105','K116'),@('K110','K111'),@('K118','K119'),@('K097','K117'))
foreach($pair in $overlapExclusions){foreach($r in $rows){if($r.intent_group_id -eq $pair[0]){$r.negative_exclusion_intent_ids+= $pair[1]};if($r.intent_group_id -eq $pair[1]){$r.negative_exclusion_intent_ids+=$pair[0]}}}
# Parse every text paragraph and every CSV record. Coverage is NOT a claim of independent semantic certification.
Add-Type -AssemblyName Microsoft.VisualBasic
$coverage=[Collections.Generic.List[object]]::new()
$evidenceBySource=@{}
foreach($r in $rows){foreach($e in $r.sources){
    if(!$evidenceBySource.ContainsKey($e.path)){$evidenceBySource[$e.path]=[Collections.Generic.List[object]]::new()}
    $evidenceBySource[$e.path].Add([pscustomobject]@{intent=$r.intent_group_id;start=$e.start_utf16;stop=$e.start_utf16+$e.length_utf16})
}}
foreach($s in $manifest.sources){
    $used=@($rows|Where-Object{$_.sources.path -contains $s.path}|ForEach-Object{$_.intent_group_id}|Select-Object -Unique)
    if($s.path.EndsWith('.csv')){
        $sr=[IO.StringReader]::new($texts[$s.path]);$parser=[Microsoft.VisualBasic.FileIO.TextFieldParser]::new($sr)
        $parser.SetDelimiters(',');$parser.HasFieldsEnclosedInQuotes=$true;$parser.TrimWhiteSpace=$false
        try{$headers=$parser.ReadFields();$n=0;while(!$parser.EndOfData){$line=$parser.LineNumber;$fields=$parser.ReadFields();$n++
            if($fields.Count -ne $headers.Count){throw "CSV width mismatch $($s.path) record $n"}
            $reason=if($s.evidence_policy -like 'excluded*'){'舊FAQ或AI整理欄位不得作事實來源；整列排除，不抽取其答案。'}elseif($s.path -match '107~114'){'名單缺乏逐人錄取證明、母體與統計口徑；不據以生成錄取率或保證。'}elseif($s.path -match '相關資料搜集'){'整理敘述缺直接原始文件定位；含招生外加名額口徑及專題時程疑義，不作規章依據。'}else{'社群紀錄逐列解析；發布日期不等於規章適用期，未有足夠原始規章定位，不作新增制度事實。'}
            $coverage.Add([pscustomobject]@{path=$s.path;source_sha256=$s.sha256;unit_type='csv_record';record=$n;physical_start_line=$line;headers=$headers;nonempty_cells=@($fields|Where-Object{$_ -and $_.Trim()}).Count;decision='not_used_as_fact_evidence';reason=$reason;semantic_certification='not_established'})
        };if($n -ne $s.csv_data_rows){throw "CSV count changed $($s.path)"}}finally{$parser.Dispose();$sr.Dispose()}
    }else{
        $t=$texts[$s.path];$n=0
        foreach($match in [regex]::Matches($t,'(?s)\S.*?(?=(?:\r?\n[\t ]*\r?\n)|\z)')){
            $n++;$matchedSet=[Collections.Generic.HashSet[string]]::new()
            foreach($e in $evidenceBySource[$s.path]){if($e.start -lt $match.Index+$match.Length -and $e.stop -gt $match.Index){[void]$matchedSet.Add($e.intent)}}
            $matched=@($matchedSet|Sort-Object)
            $reason=if($matched){'與採用的知識點證據範圍相交；精確引用另存於母資料。'}elseif($s.merge_conflict_markers.Count){'文件含未解決合併衝突，不作事實依據。'}elseif($s.evidence_policy -like 'transcript*'){'逐段解析但口稿或轉寫非獨立規章證明；未獨立確立來源歸屬與現行性，不新增政策答案。'}else{'段落已解析；屬未選用、重複、導覽、歷史公告、宣傳或需補證內容，未宣稱通過完整語意審核。'}
            $coverage.Add([pscustomobject]@{path=$s.path;source_sha256=$s.sha256;unit_type='text_paragraph';record=$n;start_utf16=$match.Index;length_utf16=$match.Length;line_start=1+[regex]::Matches($t.Substring(0,$match.Index),'\r\n|\r|\n|\u2028').Count;decision=if($matched){'intersects_selected_evidence'}else{'not_selected'};intent_ids=$matched;reason=$reason;semantic_certification='not_independently_established'})
        }
    }
    $s|Add-Member -Force candidate_intent_ids $used
    $review=@($spec.supplement.source_reviews|Where-Object{$_.path -ceq $s.path})
    if($review.Count){$s|Add-Member -Force supplement_review $review[0]}
}
# All source reads and authored evidence anchors have succeeded before replacing derived artifacts.
WriteRows (Join-Path $dir 'qa_master.jsonl') $rows
WriteRows (Join-Path $dir 'source_coverage.jsonl') $coverage
$manifest.schema_version='training_v2_sources_2'
$manifest|Add-Member -Force coverage_ledger 'source_coverage.jsonl'
$manifest|Add-Member -Force coverage_ledger_sha256 (Hash (Join-Path $dir 'source_coverage.jsonl'))
$manifest|Add-Member -Force expansion_spec_sha256 (Hash (Join-Path $dir 'expansion_spec.json'))
if($spec.supplement){$manifest|Add-Member -Force supplement_baseline $spec.supplement.baseline}
WriteJson (Join-Path $dir 'source_manifest.json') $manifest

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Esc($v){[Security.SecurityElement]::Escape([string]$v)}
$sb=[Text.StringBuilder]::new()
[void]$sb.Append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="62" customWidth="1"/><col min="2" max="2" width="104" customWidth="1"/><col min="3" max="3" width="24" customWidth="1"/></cols><sheetData>')
$sheetRows=,@('題目','答案','訓練用途');foreach($r in $rows){$sheetRows+=,@($r.question,$r.answer,($r.training_uses -join '、'))}
for($i=0;$i -lt $sheetRows.Count;$i++){
    $cells=$sheetRows[$i];$height=if($i){[Math]::Max(48,[Math]::Ceiling(([string]$cells[1]).Length/48)*20+8)}else{28}
    [void]$sb.Append('<row r="'+($i+1)+'" ht="'+$height+'" customHeight="1">')
    for($j=0;$j -lt 3;$j++){[void]$sb.Append('<c r="'+[char](65+$j)+($i+1)+'" s="'+$(if($i){1}else{2})+'" t="inlineStr"><is><t xml:space="preserve">'+(Esc $cells[$j])+'</t></is></c>')};[void]$sb.Append('</row>')
}
[void]$sb.Append('</sheetData><autoFilter ref="A1:C'+$sheetRows.Count+'"/></worksheet>')
$entries=@{
    '[Content_Types].xml'='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>';
    '_rels/.rels'='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>';
    'xl/workbook.xml'='<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="訓練問答總表" sheetId="1" r:id="rId1"/></sheets></workbook>';
    'xl/_rels/workbook.xml.rels'='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>';
    'xl/styles.xml'='<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Microsoft JhengHei"/></font><font><b/><sz val="11"/><name val="Microsoft JhengHei"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf></cellXfs></styleSheet>';
    'xl/worksheets/sheet1.xml'=$sb.ToString()
}
$dest=Join-Path $dir '訓練問答總表.xlsx'
$fs=[IO.File]::Open($dest,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::None)
$z=[IO.Compression.ZipArchive]::new($fs,[IO.Compression.ZipArchiveMode]::Create)
try{foreach($key in $entries.Keys){$entry=$z.CreateEntry($key);$writer=[IO.StreamWriter]::new($entry.Open(),$utf);try{$writer.Write($entries[$key])}finally{$writer.Dispose()}}}finally{$z.Dispose();$fs.Dispose()}
[pscustomobject]@{rows=$rows.Count;authored_expansion_points=$spec.points.Count;supplement_new_points=$spec.supplement.new_points;coverage_units=$coverage.Count;csv_records=@($coverage|Where-Object{$_.unit_type -eq 'csv_record'}).Count}|ConvertTo-Json -Compress
