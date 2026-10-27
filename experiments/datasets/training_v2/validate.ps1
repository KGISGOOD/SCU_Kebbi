param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot '../../..'),
    [string]$ReportPath = (Join-Path $PSScriptRoot 'validation_report.json'),
    [switch]$RefreshAutoMetadata,
    [switch]$RequireFormal
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot)
$dir = Join-Path $root 'experiments/datasets/training_v2'
$utf = [Text.UTF8Encoding]::new($false, $true)
$issues = [Collections.Generic.List[object]]::new()
function Issue($code, $where, $message) {
    $issues.Add([pscustomobject]@{code=$code; location=$where; message=$message})
}
function ReadText($path) { $utf.GetString([IO.File]::ReadAllBytes($path)).TrimStart([char]0xFEFF) }
function Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
function ReadRows($path) {
    $result = @(); $line = 0
    foreach ($text in [regex]::Split((ReadText $path), '\r\n|\n|\r')) {
        $line++
        if (!$text.Trim()) { continue }
        try { $result += $text | ConvertFrom-Json -ErrorAction Stop }
        catch { Issue 'json_parse' "$path/$line" $_.Exception.Message }
    }
    return $result
}
function SourcePath($relative) {
    if ($relative -notmatch '^畢業專題_爬蟲/type[12]/') { throw "Disallowed source: $relative" }
    $full = [IO.Path]::GetFullPath((Join-Path $root $relative))
    $allowed = @('type1','type2') | Where-Object {
        $full.StartsWith((Join-Path $root ('畢業專題_爬蟲/'+$_))+[IO.Path]::DirectorySeparatorChar,
                        [StringComparison]::OrdinalIgnoreCase)
    }
    if (!$allowed) { throw "Source escapes allowed roots: $relative" }
    $item = Get-Item -LiteralPath $full -Force
    while ($item -and $item.FullName.Length -ge $root.Length) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse source' }
        if ($item.PSIsContainer) { $item=$item.Parent } else { $item=$item.Directory }
    }
    return $full
}
$manifestPath = Join-Path $dir 'source_manifest.json'
$masterPath = Join-Path $dir 'qa_master.jsonl'
$manifest = ReadText $manifestPath | ConvertFrom-Json
$specPath=Join-Path $dir 'expansion_spec.json'
$spec=ReadText $specPath|ConvertFrom-Json
if((Hash $specPath) -cne $manifest.expansion_spec_sha256){Issue 'reviewed_spec_hash' 'expansion_spec.json' 'Authored question/answer specification changed after artifact build'}
$specById=@{};foreach($p in $spec.points){$specById[$p.id]=$p}
$sourceMap = @{}; $texts = @{}
foreach ($s in $manifest.sources) {
    try {
        $path=SourcePath $s.path
        if ((Hash $path) -cne $s.sha256) { Issue 'source_hash' $s.path 'Original bytes differ from source manifest' }
        $sourceMap[$s.path]=$s
        $texts[$s.path]=ReadText $path
        if ($s.parse_status -ne 'parsed') { Issue 'source_parse' $s.path 'Source inventory records a parsing failure' }
    } catch { Issue 'source_unavailable' $s.path $_.Exception.Message }
}
$actualPaths=@(Get-ChildItem -LiteralPath (Join-Path $root '畢業專題_爬蟲/type1'),(Join-Path $root '畢業專題_爬蟲/type2') -Recurse -Force -File |
    ForEach-Object {$_.FullName.Substring($root.Length+1).Replace('\','/')} | Sort-Object)
if (($actualPaths -join '|') -cne (@($manifest.sources.path|Sort-Object) -join '|')) {
    Issue 'source_inventory' 'source_manifest' 'Allowed source file inventory changed'
}
$inspectedCode=if($manifest.training_preparation_code){$manifest.training_preparation_code}else{$manifest.read_only_training_code}
foreach ($c in $inspectedCode) {
    if (!(Test-Path -LiteralPath (Join-Path $root $c.path)) -or (Hash (Join-Path $root $c.path)) -cne $c.sha256) {
        Issue 'training_code_changed' $c.path 'Read-only inspected training code changed'
    }
}
$rows=@(ReadRows $masterPath); $byId=@{}; $questionKeys=@{}; $positive=@{}; $hard=@{}
$baselineQa=@($rows|Where-Object{($_.intent_group_id -match '^K(\d+)$' -and [int]$Matches[1] -le 101) -or $_.intent_group_id -match '^M00[1-4]$'})
$corrections=@{};foreach($c in $spec.content_corrections.changes){
    if($corrections.ContainsKey($c.question_id)){Issue 'duplicate_correction' $c.question_id 'Duplicate correction'}
    $corrections[$c.question_id]=$c
}
foreach($c in $corrections.Values){
    $actual=@($rows|Where-Object question_id -eq $c.question_id)
    if($actual.Count -ne 1 -or $actual[0].question -cne $c.after_question -or $actual[0].answer -cne $c.after_answer){Issue 'correction_mismatch' $c.question_id 'Current content differs from documented after values'}
}
function HistoricalIdentity($items){
    (@($items|ForEach-Object{
        $q=$_.question;$a=$_.answer
        if($corrections.ContainsKey($_.question_id)){$q=$corrections[$_.question_id].before_question;$a=$corrections[$_.question_id].before_answer}
        [pscustomobject][ordered]@{id=$_.question_id;question=$q;answer=$a}|ConvertTo-Json -Compress
    }) -join [char]10)+[char]10
}
$identity=HistoricalIdentity $baselineQa
$sha=[Security.Cryptography.SHA256]::Create()
try{$baselineDigest=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity)))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
if($baselineQa.Count -ne $spec.supplement.baseline.baseline_count -or $baselineDigest -cne $spec.supplement.baseline.qa_identity_sha256){Issue 'baseline_qa_changed' 'qa_master.jsonl' 'The previous 261 question IDs/texts/answers or their order changed'}
if($spec.content_corrections){
    $sha=[Security.Cryptography.SHA256]::Create()
    try{$restoredDigest=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes((HistoricalIdentity $rows))))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
    if($restoredDigest -cne $spec.content_corrections.input_qa_identity_sha256 -or ($rows.question_id -join '|') -cne ($spec.content_corrections.checked_question_ids -join '|')){Issue 'correction_baseline' 'qa_master.jsonl' 'Undocumented content, ID or order change against the 336-row input'}
}
$reviews=@($spec.supplement.source_reviews)
if($reviews.Count -ne 37 -or (@($reviews.path|Sort-Object) -join '|') -cne (@($spec.supplement.baseline.paths|Sort-Object) -join '|')){Issue 'unused_source_review_inventory' 'expansion_spec.json' 'All 37 originally unreferenced files must have exactly one review'}
foreach($rv in $reviews){if(!$sourceMap.ContainsKey($rv.path) -or !$rv.reason -or $rv.status -ne 'assistant_source_screen_not_human_approval'){Issue 'unused_source_review' $rv.path 'Missing source/reason or falsely approved review'}}
foreach ($r in $rows) {
    $id=$r.question_id
    if ($id -notmatch '^TV2_[KM][0-9]{3}_[0-9]{2}$' -or $byId.ContainsKey([string]$id)) {
        Issue 'id' $id 'Invalid or duplicated question ID'; continue
    }
    $byId[$id]=$r
    if($specById.ContainsKey($r.intent_group_id)){
        $p=$specById[$r.intent_group_id]
        $variant=[int]($id.Substring($id.Length-2))-1
        if($variant -lt 0 -or $variant -ge $p.questions.Count -or $r.question -cne $p.questions[$variant] -or $r.answer -cne $p.answer){
            Issue 'authored_semantic_review_binding' $id 'Question/answer differs from the clause-and-scope reviewed candidate specification'
        }
        if($r.sources[0].path -cne ('畢業專題_爬蟲/type1/'+$p.file) -or $r.applicable_period -cne $p.scope){Issue 'reviewed_scope_binding' $id 'Source document or applicability differs from reviewed specification'}
    }
    if ($r.question -isnot [string] -or !$r.question.Trim() -or $r.answer -isnot [string] -or !$r.answer.Trim()) {
        Issue 'empty_qa' $id 'Question and answer must be nonempty strings'; continue
    }
    $key=[regex]::Replace($r.question.Normalize([Text.NormalizationForm]::FormKC),'\s+','')
    if ($questionKeys.ContainsKey($key)) { Issue 'duplicate_question' $id "Duplicates $($questionKeys[$key])" }
    $questionKeys[$key]=$id
    if ($null -ne $r.split) { Issue 'premature_split' $id 'No Train/Validation/Test assignment allowed in this mother dataset' }
    if ($r.review_status -ne 'not_human_approved' -or $r.auto_validation.formal_training_eligible -ne $false) {
        Issue 'approval_state' $id 'Candidate data cannot assert human approval/formal eligibility'
    }
    if ($r.qa_type -notin @('basic_qa','paraphrase','multi_information')) { Issue 'qa_type' $id 'Unknown QA type' }
    if (!$r.intent_group_id -or !$r.leakage_family_id -or !$r.sources.Count -or !$r.answer_claims.Count) {
        Issue 'required_metadata' $id 'Missing intent, family, evidence or claims'
    }
    foreach ($token in $r.question_invariants) {
        if (!$r.question.Contains([string]$token)) { Issue 'question_scope' $id "Required subject/scope token missing: $token" }
    }
    $quoted=@()
    foreach ($e in $r.sources) {
        if (!$sourceMap.ContainsKey($e.path)) { Issue 'evidence_source' $id 'Unregistered source'; continue }
        $s=$sourceMap[$e.path]
        if ($s.evidence_policy -match '^excluded' -or $s.merge_conflict_markers.Count -or $e.path -match 'FAQ|ai_clean') {
            Issue 'unsafe_source' $id 'Old FAQ, AI answer or unresolved merge source cannot support facts'
        }
        if ($e.sha256 -cne $s.sha256) { Issue 'evidence_hash' $id 'Evidence and source manifest hashes differ' }
        $text=$texts[$e.path]
        if ($e.start_utf16 -lt 0 -or $e.length_utf16 -le 0 -or $e.start_utf16+$e.length_utf16 -gt $text.Length) {
            Issue 'evidence_range' $id 'Invalid UTF-16 source range'; continue
        }
        if ($text.Substring($e.start_utf16,$e.length_utf16) -cne $e.quote) { Issue 'evidence_quote' $id 'Quote is not exact source text at stored offset' }
        $startLine=1+[regex]::Matches($text.Substring(0,$e.start_utf16),'\r\n|\r|\n|\u2028').Count
        $endLine=1+[regex]::Matches($text.Substring(0,$e.start_utf16+$e.length_utf16),'\r\n|\r|\n|\u2028').Count
        if ($startLine -ne $e.line_start -or $endLine -ne $e.line_end) { Issue 'evidence_lines' $id 'Line locator mismatch' }
        $quoted+=[string]$e.quote
    }
    $allQuotes=$quoted -join [char]10
    foreach ($phrase in $r.required_answer_phrases) {
        if (!$r.answer.Contains([string]$phrase) -or
            !([regex]::Replace($allQuotes,'\s+','')).Contains([regex]::Replace([string]$phrase,'\s+',''))) {
            Issue 'required_answer_condition' $id "Required qualifier/prerequisite omitted: $phrase"
        }
    }
    foreach ($claim in $r.answer_claims) {
        if (!$r.answer.Contains([string]$claim.value)) { Issue 'claim_answer' $id 'Declared literal claim missing from answer' }
        foreach ($i in $claim.evidence_indexes) {
            if ($i -lt 0 -or $i -ge $r.sources.Count -or !$r.sources[$i].quote.Contains([string]$claim.value)) {
                Issue 'claim_evidence' $id "Claim literal absent from assigned quote: $($claim.value)"
            }
        }
        if (!$claim.evidence_indexes.Count) { Issue 'claim_evidence' $id 'No supporting quote index' }
    }
    # Literal number checks detect invented values, not swapped semantic roles or arbitrary entailment.
    foreach ($n in [regex]::Matches($r.answer,'\d+(?:,\d{3})*(?:\.\d+)?')) {
        if (!$allQuotes.Contains($n.Value)) { Issue 'unsupported_number' $id "Number absent from evidence: $($n.Value)" }
    }
    if ($null -ne $r.academic_year) {
        $year=[string]$r.academic_year
        if (!$r.question.Contains($year) -or !$r.answer.Contains($year) -or !$allQuotes.Contains($year)) {
            Issue 'year_scope' $id 'Academic year must appear in question, answer and evidence'
        }
    }
    foreach ($pair in $r.positive_pairs) {
        if ($positive.ContainsKey($pair.pair_id) -and ($positive[$pair.pair_id]|ConvertTo-Json -Compress) -cne ($pair|ConvertTo-Json -Compress)) {
            Issue 'positive_pair_definition' $id 'Conflicting copies of a positive pair'
        }
        $positive[$pair.pair_id]=$pair
    }
    foreach ($pair in $r.hard_negative_pairs) {
        if ($hard.ContainsKey($pair.pair_id) -and ($hard[$pair.pair_id]|ConvertTo-Json -Compress) -cne ($pair|ConvertTo-Json -Compress)) {
            Issue 'hard_pair_definition' $id 'Conflicting copies of a hard-negative pair'
        }
        $hard[$pair.pair_id]=$pair
    }
}
foreach ($p in $positive.Values) {
    if (!$byId.ContainsKey($p.query_id) -or !$byId.ContainsKey($p.positive_id) -or $p.query_id -eq $p.positive_id) {
        Issue 'positive_reference' $p.pair_id 'Invalid/self pair'; continue
    }
    $a=$byId[$p.query_id];$b=$byId[$p.positive_id]
    if ($a.intent_group_id -cne $b.intent_group_id -or $a.answer -cne $b.answer -or
        ($a.intent_signature|ConvertTo-Json -Compress) -cne ($b.intent_signature|ConvertTo-Json -Compress)) {
        Issue 'positive_contract' $p.pair_id 'Intent signature, answer or scope differs'
    }
}
foreach ($p in $hard.Values) {
    if (!$byId.ContainsKey($p.anchor_id) -or !$byId.ContainsKey($p.negative_id)) { Issue 'negative_reference' $p.pair_id 'Missing endpoint'; continue }
    $a=$byId[$p.anchor_id];$b=$byId[$p.negative_id]
    if ($a.intent_group_id -eq $b.intent_group_id -or $a.question -ceq $b.question -or $a.answer -ceq $b.answer) {
        Issue 'self_or_equal_negative' $p.pair_id 'Same intent/text/answer cannot be a negative'
    }
    if($a.negative_exclusion_intent_ids -contains $b.intent_group_id -or $b.negative_exclusion_intent_ids -contains $a.intent_group_id){Issue 'excluded_negative_pair' $p.pair_id 'An overlapping parent/child request was incorrectly used as hard negative'}
    if ($a.leakage_family_id -cne $b.leakage_family_id -or !$p.rationale -or $p.model_difficulty -ne 'not_measured') {
        Issue 'negative_contract' $p.pair_id 'Requires shared topic, explicit contrast rationale and honest unmeasured status'
    }
}
foreach ($r in $rows) {
    foreach($otherIntent in $r.negative_exclusion_intent_ids){
        $others=@($rows|Where-Object intent_group_id -ceq $otherIntent)
        if(!$others.Count -or @($others|Where-Object{$_.negative_exclusion_intent_ids -notcontains $r.intent_group_id}).Count){Issue 'negative_exclusion_reference' $r.question_id 'Missing/asymmetric overlap exclusion'}
    }
    foreach ($other in $r.paraphrase_ids) {
        if (!$byId.ContainsKey($other) -or $byId[$other].intent_group_id -cne $r.intent_group_id -or
            $other -eq $r.question_id -or $byId[$other].paraphrase_ids -notcontains $r.question_id) {
            Issue 'paraphrase_reference' $r.question_id 'Invalid/asymmetric synonym reference'
        }
    }
    if (!$byId.ContainsKey($r.canonical_question_id)) { Issue 'canonical_reference' $r.question_id 'Missing canonical question' }
    $expectedUses=@();if (@($positive.Values|Where-Object {$_.query_id -eq $r.question_id -or $_.positive_id -eq $r.question_id}).Count) {
        $expectedUses+=@('A2-1','A2-2')
    }
    $expectedUses+='A3'
    if (($r.training_uses -join '|') -cne ($expectedUses -join '|')) { Issue 'training_use' $r.question_id 'Usage must reflect actual pair membership' }
    if ($r.qa_type -eq 'multi_information') {
        if ($r.knowledge_point_ids.Count -lt 2 -or $r.sources.Count -lt 2 -or $r.training_uses -contains 'A2-1') {
            Issue 'integration_contract' $r.question_id 'Integration needs multiple components/evidence; excluded from embedding in this version'
        }
    }
}
# Bounded conflict checks: do not call this an exhaustive cross-document contradiction proof.
$conflicts=@()
foreach ($s in $manifest.sources) { if ($s.merge_conflict_markers.Count) {
    $conflicts+=[pscustomobject]@{kind='merge_conflict';path=$s.path;excluded_as_evidence=$true}
} }
$course=$texts['畢業專題_爬蟲/type1/課程資訊_學士班.txt']
if ($course -match '最多可承認外系選修16學分' -and $course -match '若有修讀輔系或跨領域學分學程，可承認10學分') {
    $conflicts+=[pscustomobject]@{kind='scope_sensitive_external_credit_cap';path='畢業專題_爬蟲/type1/課程資訊_學士班.txt';excluded_as_universal_answer=$true}
}
foreach ($group in ($rows|Group-Object intent_group_id)) {
    if (@($group.Group.answer|Select-Object -Unique).Count -ne 1) { Issue 'intra_intent_conflict' $group.Name 'One intent has different answers' }
}
$nearDuplicates=@()
function ContentKey($text){[regex]::Replace($text.Normalize([Text.NormalizationForm]::FormKC),'[\s\p{P}\p{S}]','')}
foreach($g in ($rows|Group-Object intent_group_id)){
    $variants=@($g.Group)
    for($i=0;$i -lt $variants.Count;$i++){for($j=$i+1;$j -lt $variants.Count;$j++){
        $a=ContentKey $variants[$i].question;$b=ContentKey $variants[$j].question
        if($a -ceq $b){Issue 'punctuation_only_paraphrase' $variants[$j].question_id 'Same content after punctuation/spacing normalization'}
        $aSet=[Collections.Generic.HashSet[string]]::new();$bSet=[Collections.Generic.HashSet[string]]::new()
        for($k=0;$k -le $a.Length-3;$k++){[void]$aSet.Add($a.Substring($k,3))}
        for($k=0;$k -le $b.Length-3;$k++){[void]$bSet.Add($b.Substring($k,3))}
        $union=[Collections.Generic.HashSet[string]]::new($aSet);$union.UnionWith($bSet)
        $intersection=[Collections.Generic.HashSet[string]]::new($aSet);$intersection.IntersectWith($bSet)
        if($union.Count -and $intersection.Count/$union.Count -gt 0.85){$nearDuplicates+=[pscustomobject]@{left=$variants[$i].question_id;right=$variants[$j].question_id;trigram_jaccard=$intersection.Count/$union.Count;status='needs_review_not_automatically_deduplicated'}}
    }}
}
# Bounded role/logic checks: more than numeric co-occurrence, still not a general entailment oracle.
$logicRules=@{
    K033=@('抵免是.*計入畢業學分','免修.*不計入畢業學分.*補足');
    K037=@('超過十年.*不得抵免','碩士在職專班不受');
    K038=@('以多抵少.*以少學分');
    K039=@('不能補修.*不予列抵','可以補修.*下學期','停開.*學系主任');
    K040=@('轉學新生.*入學輔導','其他.*加退選截止日前','一次為原則','追抵.*第三條');
    K047=@('每學年上學期.*一次','名額視當年財源','學生.*二萬元.*專任教師.*五萬元.*為原則');
    K058=@('學分學程至少十五學分','微學程至少六學分','至少二分之一.*不屬於.*必修');
    K061=@('二分之一為上限.*至多十學分');
    K063=@('經系上審核通過','在學期間.*至少160小時');
    K067=@('爭取.*不等於');
    K068=@('超過160小時.*只有2學分','依合約.*薪資.*保障');
    K069=@('只在寒暑假.*不能','學期中.*160小時','持續到學期末.*可選修');
    K073=@('四年級.*延修生.*計畫出國交換.*三年級','至少一科須及格');
    K077=@('核定通過.*才可個人','其餘組員維持原編制');
    K084=@('不得任意換班.*特殊規定');
    K085=@('僅修一學期.*或.*不及格.*不承認');
    K093=@('112學年度.*大一下學期.*3學分','程式設計（一）須60分以上')
    K118=@('2025年.*學士生先修碩士班課程','AMP');
    K121=@('宋浩.*機器學習','未註明適用學期');
    K122=@('劉豐泰.*資料庫管理與實務','未註明適用學期');
    K123=@('蕭宇程.*區塊鏈','未註明適用學期');
    K128=@('課程名稱完全相同','學分高於或等於','若本院有演習課.*他系課程也須有演習課','主任審核');
    K129=@('112學年度.*恢復','不能|不據以');
    K130=@('借還圖書.*5136.*5137','非書服務5139','期刊服務5138');
    K131=@('借還圖書2449','參考諮詢與館際合作2445或2446','期刊2442','非書服務2443或2444')
}
foreach($r in $rows){if($logicRules.ContainsKey($r.intent_group_id)){foreach($rule in $logicRules[$r.intent_group_id]){if($r.answer -notmatch $rule){Issue 'bounded_answer_role_logic' $r.question_id "Lost role/condition relation: $rule"}}}}
$coveragePath=Join-Path $dir 'source_coverage.jsonl'
$coverage=@(ReadRows $coveragePath)
if((Hash $coveragePath) -cne $manifest.coverage_ledger_sha256){Issue 'coverage_hash' 'source_coverage.jsonl' 'Coverage ledger changed'}
foreach($s in $manifest.sources){
    $units=@($coverage|Where-Object{$_.path -ceq $s.path})
    if(!$units.Count){Issue 'coverage_missing' $s.path 'No paragraph/record inventory'}
    if($s.path.EndsWith('.csv') -and $units.Count -ne $s.csv_data_rows){Issue 'csv_coverage_count' $s.path 'Not every parsed CSV record is accounted for'}
    $expected=1
    foreach($u in $units){
        if($u.source_sha256 -cne $s.sha256 -or $u.record -ne $expected){Issue 'coverage_identity' $s.path 'Wrong source hash or nonconsecutive unit number'}
        $expected++
        if($u.unit_type -eq 'text_paragraph' -and ($u.start_utf16 -lt 0 -or $u.length_utf16 -le 0 -or $u.start_utf16+$u.length_utf16 -gt $texts[$s.path].Length)){Issue 'coverage_range' $s.path 'Invalid paragraph range'}
    }
}
$excelOk=$false
try {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z=[IO.Compression.ZipFile]::OpenRead((Join-Path $dir '訓練問答總表.xlsx'))
    try {
        function XmlEntry($name) {
            $rd=[IO.StreamReader]::new($z.GetEntry($name).Open(),$utf)
            try { [xml]$rd.ReadToEnd() } finally { $rd.Dispose() }
        }
        $wb=XmlEntry 'xl/workbook.xml'
        if (@($wb.workbook.sheets.sheet).Count -ne 1) { throw 'Expected one worksheet' }
        $ws=XmlEntry 'xl/worksheets/sheet1.xml';$xr=@($ws.worksheet.sheetData.row)
        if ($xr.Count -ne $rows.Count+1) { throw 'Workbook row count differs' }
        $shared=$null;if($z.GetEntry('xl/sharedStrings.xml')){$shared=XmlEntry 'xl/sharedStrings.xml'}
        function CellText($cell) {
            if ($cell.t -eq 's') { return $shared.sst.si[[int]$cell.v].InnerText }
            if ($cell.t -eq 'inlineStr') { return $cell.is.InnerText }
            return [string]$cell.v
        }
        for ($i=0;$i -lt $xr.Count;$i++) {
            if (@($xr[$i].c).Count -ne 3) { throw 'Workbook must have exactly three cells per row' }
            $want=@('題目','答案','訓練用途')
            if($i){$want=@($rows[$i-1].question,$rows[$i-1].answer,($rows[$i-1].training_uses -join '、'))}
            for($j=0;$j -lt 3;$j++){
                if((CellText $xr[$i].c[$j]) -cne [string]$want[$j]){throw "Workbook/master mismatch row $($i+1) column $($j+1)"}
            }
        }
        $excelOk=$true
    } finally { $z.Dispose() }
} catch { Issue 'excel_master_consistency' '訓練問答總表.xlsx' $_.Exception.Message }
$excluded=@(ReadRows (Join-Path $dir 'excluded_candidates.jsonl'))
foreach($x in $excluded){
    if($x.answer -or $x.formal_training_eligible -ne $false -or !$x.reason_code){Issue 'exclusion_contract' $x.draft_id 'Excluded drafts cannot contain training answers/approval'}
    foreach($s in $x.sources){if(!$sourceMap.ContainsKey($s.path) -or $sourceMap[$s.path].sha256 -cne $s.sha256){Issue 'exclusion_source' $x.draft_id 'Unregistered/changed exclusion evidence'}}
}
if($RefreshAutoMetadata -and !$issues.Count){
    foreach($r in $rows){
        $r.auto_validation.evidence_checks='exact_spans_hashes_claim_literals_passed'
        $r.auto_validation.conditions_checks='year_subject_tokens_and_numeric_literals_passed_not_full_entailment'
    }
    [IO.File]::WriteAllText($masterPath,(($rows|ForEach-Object {$_|ConvertTo-Json -Depth 20 -Compress}) -join [char]10)+[char]10,[Text.UTF8Encoding]::new($false))
}
$report=[pscustomobject][ordered]@{
    schema_version='training_v2_validation_2';
    status=if($issues.Count){'failed_deterministic_checks'}else{'deterministic_checks_passed_semantics_unverified'};
    errors_count=$issues.Count;errors=@($issues.ToArray());
    source_files_read=$manifest.sources.Count;source_files_unchanged=(@($issues|Where-Object {$_.code -match '^source_'}).Count -eq 0);
    source_files_with_selected_evidence=@($rows.sources.path|Select-Object -Unique).Count;
    source_coverage_units=$coverage.Count;csv_records_accounted_for=@($coverage|Where-Object{$_.unit_type -eq 'csv_record'}).Count;
    text_paragraphs_accounted_for=@($coverage|Where-Object{$_.unit_type -eq 'text_paragraph'}).Count;
    coverage_meaning='All source bytes parsed and units inventoried; unselected paragraphs/CSV rows are not claimed semantically certified';
    master_rows=$rows.Count;knowledge_groups=@($rows.intent_group_id|Select-Object -Unique).Count;
    independent_atomic_knowledge_points=@($rows|Where-Object{$_.qa_type -eq 'basic_qa'}).Count;
    previous_261_qa_unchanged=($baselineDigest -ceq $spec.supplement.baseline.qa_identity_sha256 -and $baselineQa.Count -eq 261 -and !@($baselineQa|Where-Object{$corrections.ContainsKey($_.question_id)}).Count);
    historical_baseline_reconciled_by_documented_corrections=($baselineDigest -ceq $spec.supplement.baseline.qa_identity_sha256);
    content_review=[pscustomobject]@{checked_rows=@($spec.content_corrections.checked_question_ids).Count;changed_rows=$corrections.Count;question_changes=@($corrections.Values|Where-Object{$_.before_question -cne $_.after_question}).Count;answer_changes=@($corrections.Values|Where-Object{$_.before_answer -cne $_.after_answer}).Count;review_status='assistant_source_review_not_human_approval';unresolved=$spec.content_corrections.unresolved};
    supplement=[pscustomobject]@{reviewed_previous_unused_files=$reviews.Count;new_basic_questions=@($rows|Where-Object{$_.intent_group_id -in $spec.supplement.new_point_ids -and $_.qa_type -eq 'basic_qa'}).Count;new_paraphrases=@($rows|Where-Object{$_.intent_group_id -in $spec.supplement.new_point_ids -and $_.qa_type -eq 'paraphrase'}).Count;newly_referenced_files=@($manifest.sources|Where-Object{$_.path -in $spec.supplement.baseline.paths -and $_.candidate_intent_ids.Count}).Count;still_unreferenced_files=@($manifest.sources|Where-Object{$_.path -in $spec.supplement.baseline.paths -and !$_.candidate_intent_ids.Count}).Count};
    topics=@($rows|Group-Object topic|Select-Object Name,Count);
    content_correction_counts=[pscustomobject]@{
        unchanged_question_and_answer_rows=$rows.Count-$corrections.Count;
        revised_paraphrase_rows_including_shared_answer_fixes=@($rows|Where-Object{$_.qa_type -eq 'paraphrase' -and $corrections.ContainsKey($_.question_id)}).Count;
        affected_paraphrase_groups=@($rows|Where-Object{$_.qa_type -eq 'paraphrase' -and $corrections.ContainsKey($_.question_id)}|Select-Object -ExpandProperty intent_group_id -Unique).Count;
        affected_positive_pairs=@($positive.Values|Where-Object{$corrections.ContainsKey($_.query_id) -or $corrections.ContainsKey($_.positive_id)}).Count;
        newly_excluded_rows=@($spec.content_corrections.removed_question_ids).Count
    };
    types=@($rows|Group-Object qa_type|Select-Object Name,Count);
    deterministic_checks_passed_rows=if($issues.Count){$null}else{$rows.Count};
    positive_pairs=$positive.Count;hard_negative_candidate_pairs=$hard.Count;
    hard_negative_participant_questions=@($hard.Values|ForEach-Object {$_.anchor_id;$_.negative_id}|Select-Object -Unique).Count;
    overlapping_intent_exclusion_pairs=@($rows|ForEach-Object{$a=$_.intent_group_id;foreach($b in $_.negative_exclusion_intent_ids){(@($a,$b)|Sort-Object)-join '|'}}|Select-Object -Unique).Count;
    uses=@();
    excluded_draft_rows=$excluded.Count;excluded_reasons=@($excluded|Group-Object reason_code|Select-Object Name,Count);
    bounded_source_risks=$conflicts;excel_jsonl_identical=$excelOk;
    split_created=$false;human_approval_created=$false;formal_training_eligible_rows=0;
    semantic_equivalence_independently_verified_pairs=0;hard_negative_semantics_independently_verified_pairs=0;
    semantic_pending_rows=$rows.Count;undated_source_scope_rows=@($rows|Where-Object {$_.temporal_status -eq 'undated_snapshot_not_current_policy_certified'}).Count;
    assistant_source_scope_reviewed_candidate_rows=@($rows|Where-Object{$_.semantic_source_review}).Count;
    independent_semantic_entailment_verified_rows=0;
    bounded_role_logic_checks='credit direction, minima versus caps, prerequisites, qualifiers, time and eligibility branches';
    near_duplicate_policy='Exact normalized questions rejected; same-intent paraphrases retained deliberately. Different scopes are not deduplicated just by similarity. No embedding-based global near-duplicate claim.';
    high_lexical_similarity_variant_pairs=$nearDuplicates;
    limitations=@('Exact evidence and literal checks do not prove arbitrary natural-language entailment or global factual truth',
      'Paraphrase and hard-negative intent rationales are assistant-authored candidates, not independent semantic verification',
      'No model-based hard-negative difficulty ranking, tokenization or training has been run',
      'No live website/complete document-version verification; no use of old FAQ answers',
      'All candidates are excluded from formal training eligibility until independent semantic/source assurance is available');
    artifacts=[pscustomobject]@{qa_master_sha256=(Hash $masterPath);source_manifest_sha256=(Hash $manifestPath);excel_sha256=(Hash (Join-Path $dir '訓練問答總表.xlsx'));excluded_sha256=(Hash (Join-Path $dir 'excluded_candidates.jsonl'));validator_sha256=(Hash $PSCommandPath)}
}
# Count uses with an explicit outer variable (avoid Where-Object's nested $_ shadowing).
$report.uses=@();foreach($model in @('A2-1','A2-2','A3')){$report.uses+=[pscustomobject]@{model=$model;candidate_question_rows=@($rows|Where-Object {$_.training_uses -contains $model}).Count}}
$reportFull=[IO.Path]::GetFullPath($ReportPath)
if(!$reportFull.StartsWith($dir+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($reportFull) -ne '.json'){
    throw 'Validation report must be a JSON file inside training_v2'
}
if($reportFull -in @($masterPath,$manifestPath,(Join-Path $dir 'excluded_candidates.jsonl'))){throw 'Cannot overwrite input with validation report'}
[IO.File]::WriteAllText($reportFull,($report|ConvertTo-Json -Depth 16)+[char]10,[Text.UTF8Encoding]::new($false))
Write-Output ($report|Select-Object status,errors_count,master_rows,positive_pairs,hard_negative_candidate_pairs,excel_jsonl_identical,formal_training_eligible_rows|ConvertTo-Json -Compress)
if($issues.Count){exit 1}
if($RequireFormal){Write-Error 'Formal training is blocked: independent semantic/source verification is not established.' -ErrorAction Continue;exit 2}
