import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Camera, Check, History, Image as ImageIcon, RotateCcw, Volume2, Trash2, X, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { analyze } from '@/inference';
import hero from '@/assets/farmer-hero.jpg';
import introPortrait from '@/assets/intro-farmer-portrait.jpg';
import tomato from '@/assets/tomato.jpg';
import olive from '@/assets/olive.jpg';
import potato from '@/assets/potato.jpg';

type Crop = 'طماطم' | 'زيتون' | 'بطاطا';
type Analysis = { status: 'healthy' | 'diseased' | 'unclear'; title: string; description: string; classId?: string };
type RecordItem = { id: string; crop: Crop; image: string; analysis: Analysis; date: string };
type Screen = 'home' | 'history' | 'camera' | 'gallery' | 'result';
const crops: { name: Crop; image: string }[] = [
  { name: 'طماطم', image: tomato }, { name: 'زيتون', image: olive }, { name: 'بطاطا', image: potato },
];
const historyKey = 'fellah-ai-history';

export const Route = createFileRoute('/')({
  head: () => ({ links: [{ rel: "manifest", href: "/manifest.webmanifest" }], meta: [
    { title: 'Fellah AI | فلاح - فحص أوراق المحاصيل' },
    { name: 'description', content: 'صوّر ورقة محصولك واكتشف حالتها مع فلاح، مساعدك الفلاحي الذكي.' },
    { property: 'og:title', content: 'Fellah AI | فلاح' },
    { property: 'og:description', content: 'صوّر ورقة محصولك واكتشف حالتها مع فلاح.' },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary_large_image' },
  ] }),
  component: Index,
});

async function prepareImage(file: File): Promise<string> {
  if (!file.type.startsWith("image/")) throw new Error("اختار صورة من هاتفك.");

  const image = new Image();
  const url = URL.createObjectURL(file);

  try {
    image.src = url;
    await image.decode();

    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1200 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);

    const context = canvas.getContext("2d");
    if (!context) throw new Error("ما قدرناش نقراو الصورة.");

    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.76);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function Index() {
  const [showIntro, setShowIntro] = useState(true);
  const [crop, setCrop] = useState<Crop>('طماطم');
  const [screen, setScreen] = useState<Screen>('home');
  const [source, setSource] = useState<'camera' | 'gallery'>('gallery');
  const [photo, setPhoto] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [records, setRecords] = useState<RecordItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    try { setRecords(JSON.parse(localStorage.getItem(historyKey) || '[]')); } catch { /* ignore damaged local records */ }
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowIntro(false), 5000);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => () => { streamRef.current?.getTracks().forEach(track => track.stop()); window.speechSynthesis?.cancel(); }, []);
  useEffect(() => { if ("serviceWorker" in navigator) { window.addEventListener("load", () => { void navigator.serviceWorker.register("/sw.js"); }); } }, []);

  const saveRecords = (items: RecordItem[]) => {
    setRecords(items);
    try { localStorage.setItem(historyKey, JSON.stringify(items)); } catch { /* device storage may be full */ }
  };
  const stopCamera = () => { streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null; };
  const goHome = () => { stopCamera(); window.speechSynthesis?.cancel(); setScreen('home'); setPhoto(null); setAnalysis(null); setError(''); setLoading(false); };
  const openCamera = async () => {
    setError(''); setSource('camera'); setPhoto(null); setAnalysis(null); setScreen('camera');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
    } catch { setError('ما قدرناش نفتحوا الكاميرا. تأكد من السماح بالوصول ليها أو اختار صورة من الهاتف.'); }
  };
  const buildAnalysis = async (aiResult: Awaited<ReturnType<typeof analyze>>): Promise<Analysis> => {
    const knowledge = await fetch('/knowledge.json').then((response) => response.json());
    const topResult = aiResult.results[0];

    if (!topResult || !aiResult.confident) {
      return {
        status: 'unclear',
        title: 'الصورة ما هيش واضحة',
        description: 'عاود صوّر الورقة من قريب، وخلي الإضاءة مليحة.'
      };
    }

    const info = knowledge[topResult.classId];

    if (!info) {
      return {
        status: 'unclear',
        title: 'ما قدرناش نحدد الحالة',
        description: 'عاود صوّر الورقة وحاول تكون الصورة واضحة.'
      };
    }

    return {
      status: info.status === 'healthy' ? 'healthy' : 'diseased',
      title: info.name_darija,
      description: info.audio_text_darija,
      classId: topResult.classId
    };
  };

  const scan = async (image: string, imageSource: 'camera' | 'gallery') => {
    stopCamera(); setPhoto(image); setSource(imageSource); setAnalysis(null); setError(''); setLoading(true); setScreen('result');
    try {
      const aiResult = await analyze(image);
      const result = await buildAnalysis(aiResult);
      const item: RecordItem = { id: crypto.randomUUID(), crop, image, analysis: result as Analysis, date: new Date().toISOString() };
      console.log("FELLAH AUDIO CLASS:", (result as Analysis).classId); console.log("FELLAH AUDIO CLASS:", (result as Analysis).classId); setAnalysis(result as Analysis);
      saveRecords([item, ...records].slice(0, 12));
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'تعذّر تحليل الصورة. حاول مرة أخرى.'); }
    finally { setLoading(false); }
  };
  const onFile = async (file?: File) => {
    if (!file) return;
    try { await scan(await prepareImage(file), 'gallery'); } catch (caught) { setError(caught instanceof Error ? caught.message : 'ما قدرناش نقراو الصورة.'); }
  };
  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1200 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
    void scan(canvas.toDataURL('image/jpeg', 0.76), 'camera');
  };
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const speak = () => {
    if (!analysis?.classId) return;
    const audio = audioRef.current ?? new Audio(); audio.src = `/audio/${analysis.classId}.mp3`;
    audioRef.current = audio;
    void audio.play().catch((error) => console.error("FELLAH AUDIO ERROR:", error));
  };

// Auto-speak removed: browsers block audio without user interaction.
  // Audio plays when user clicks the "اسمع" button.
  return (
    <>
    <main className="app-shell" dir="rtl">
      <input ref={fileRef} type="file" accept="image/*" className="hidden" aria-label="اختر صورة ورقة من الهاتف" onChange={event => { void onFile(event.target.files?.[0]); event.target.value = ''; }} />
      {screen === 'home' && <>
        <div className="hero-image">
          <img src={hero} alt="فلاح يفحص النباتات في الحقل" width={1536} height={864} className="h-full w-full object-cover object-center" />
          <Button type="button" variant="secondary" size="sm" title="سجل التحليلات" aria-label="سجل التحليلات" onClick={() => setScreen('history')} className="absolute top-4 left-4 z-10 h-10 w-10 rounded-full bg-card text-primary shadow-md hover:bg-accent"><History size={21} /></Button>
          <div className="absolute bottom-4 right-5 left-5 z-10 text-primary-foreground">
            <h1 className="text-[29px] font-black leading-tight" dir="ltr">Fellah AI</h1>
            <p className="text-[14px] font-semibold">صوّر الورقة تاع نبتتك ونقولك واش بيها</p>
          </div>
        </div>
        <div className="px-5 pt-9 pb-10">
          <div className="mb-4 flex items-center gap-3"><span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-base font-black text-primary-foreground">1</span><h2 className="text-xl font-bold">اختار المحصول</h2></div>
          <div className="grid grid-cols-3 gap-3">
            {crops.map(item => <Button key={item.name} type="button" variant="ghost" aria-pressed={crop === item.name} onClick={() => setCrop(item.name)} data-selected={crop === item.name} className="crop-card !h-auto !w-full !p-0 flex-col gap-0 whitespace-normal hover:bg-card" title={`اختار ${item.name}`}>
              <img src={item.image} alt={item.name} loading="lazy" width={816} height={816} className="aspect-square w-full object-cover" />
              <span className="flex h-10 w-full items-center justify-center text-[15px] font-bold text-foreground">{item.name}</span>
            </Button>)}
          </div>
          <div className="mt-8 mb-4 flex items-center gap-3"><span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-base font-black text-primary-foreground">2</span><h2 className="text-xl font-bold">صوّر الورقة المريضة</h2></div>
          <div className="flex flex-col gap-3">
            <Button type="button" variant="scan" onClick={() => void openCamera()} className="w-full"><Camera className="!h-6 !w-6" /> صوّر مباشرة</Button>
            <Button type="button" variant="scanOutline" onClick={() => { setSource('gallery'); setError(''); setScreen('gallery'); }} className="w-full"><ImageIcon className="!h-6 !w-6" /> من صور الهاتف</Button>
          </div>
          {error && <p role="alert" className="mt-4 text-center text-sm text-destructive">{error}</p>}
        </div>
      </>}
      {screen === 'gallery' && <div className="min-h-dvh bg-paper">
        <div className="flex items-center gap-4 bg-primary px-5 py-4 text-primary-foreground">
          <Button type="button" variant="ghost" size="icon" aria-label="رجوع للرئيسية" onClick={goHome} className="rounded-full bg-foreground/20 text-primary-foreground hover:bg-foreground/30"><ArrowRight /></Button>
          <h1 className="text-xl font-bold">من صور الهاتف</h1>
        </div>
        <div className="p-5">
          <Button type="button" variant="scanOutline" onClick={() => fileRef.current?.click()} className="w-full"><ImageIcon className="!h-6 !w-6" /> اختار صورة من الهاتف</Button>
          {records.length > 0 && <><h2 className="mt-8 mb-4 text-lg font-bold">الصور الأخيرة</h2><div className="grid grid-cols-3 gap-2">{records.map(item => <Button key={item.id} type="button" variant="ghost" title={`حلّل صورة ${item.crop} مرة أخرى`} onClick={() => void scan(item.image, 'gallery')} className="aspect-square h-auto w-full overflow-hidden rounded-lg p-0"><img src={item.image} alt={`صورة ${item.crop} سابقة`} className="h-full w-full object-cover" /></Button>)}</div></>}
          {error && <p role="alert" className="mt-4 text-center text-sm text-destructive">{error}</p>}
        </div>
      </div>}
      {screen === 'camera' && <div className="relative flex min-h-dvh flex-col bg-foreground text-primary-foreground">
        <div className="flex items-center justify-between p-5"><span className="text-sm font-bold">حطّ الورقة وسط المربع</span><Button type="button" variant="ghost" size="icon" aria-label="إغلاق الكاميرا" onClick={goHome} className="text-primary-foreground"><X /></Button></div>
        <div className="relative flex flex-1 items-center justify-center overflow-hidden">
          <video ref={videoRef} playsInline muted autoPlay className="absolute inset-0 h-full w-full object-cover" />
          <div className="scan-frame relative z-10 aspect-[4/5] w-[70%]" />
          {error && <div className="absolute inset-x-5 bottom-5 z-20 rounded-lg bg-card p-4 text-center text-sm text-destructive">{error}<Button type="button" variant="outline" onClick={() => { stopCamera(); setScreen('gallery'); }} className="mt-3 w-full">من صور الهاتف</Button></div>}
        </div>
        <div className="flex justify-center py-5"><Button type="button" aria-label="التقط صورة" title="التقط صورة" onClick={capture} disabled={!!error} className="h-16 w-16 rounded-full border-[5px] border-card bg-card shadow-[inset_0_0_0_3px_var(--color-foreground)] hover:bg-muted" /></div>
      </div>}
      {screen === 'result' && <div className="min-h-dvh bg-paper">
        {photo && <img src={photo} alt={`ورقة ${crop} المختارة`} className="result-image" />}
        <div className="px-5 py-5">
          {loading ? <div className="flex flex-col items-center gap-5 py-16"><div className="spinner" /><p className="text-muted-foreground">راني نحلّل الورقة...</p></div> : analysis ? <>
            <div className="mb-4 flex justify-start"><span className={`inline-flex items-center gap-1 rounded-full px-4 py-2 text-sm font-bold ${analysis.status === 'healthy' ? 'bg-success text-success-foreground' : analysis.status === 'diseased' ? 'bg-destructive text-destructive-foreground' : 'bg-secondary text-secondary-foreground'}`}><Check size={16} />{analysis.status === 'healthy' ? 'صحيحة' : analysis.status === 'diseased' ? 'تحتاج عناية' : 'غير واضحة'}</span></div>
            <h2 className="mb-2 text-[25px] font-black">{analysis.title}</h2>
            <Button type="button" variant="scan" onClick={speak} className="mb-3 w-full"><Volume2 className="!h-6 !w-6" /> اسمع</Button>
            <Button type="button" variant="scanOutline" onClick={() => { if (source === 'camera') void openCamera(); else setScreen('gallery'); }} className="mb-3 w-full"><RotateCcw className="!h-5 !w-5" /> صوّر وحدة أخرى</Button>
            <Button type="button" variant="scanOutline" onClick={() => setScreen('history')} className="w-full"><History className="!h-5 !w-5" /> شوف السجل</Button>
          </> : <div className="py-12 text-center"><AlertCircle className="mx-auto mb-3 text-destructive" /><p role="alert" className="mb-5 text-destructive">{error}</p><Button type="button" onClick={() => photo && void scan(photo, source)}>عاود المحاولة</Button></div>}
          <Button type="button" variant="ghost" onClick={goHome} className="mt-5 w-full text-primary"><ArrowRight /> رجوع للرئيسية</Button>
        </div>
      </div>}
      {screen === 'history' && <div className="min-h-dvh bg-paper">
        <div className="flex items-center justify-between bg-primary px-4 py-4 text-primary-foreground">
          <Button type="button" variant="ghost" size="icon" aria-label="رجوع للرئيسية" title="رجوع للرئيسية" onClick={goHome} className="rounded-full bg-foreground/20 text-primary-foreground hover:bg-foreground/30"><ArrowRight /></Button>
          <div className="text-right"><h1 className="text-xl font-bold">السجل</h1><p className="text-xs">كل تحليل تاع الورقة محفوظ هنا</p></div>
          <Button type="button" variant="ghost" size="sm" disabled={!records.length} onClick={() => { if (window.confirm('تمسح كامل السجل؟')) saveRecords([]); }} className="rounded-full border border-primary-foreground/50 text-xs text-primary-foreground hover:bg-foreground/20">مسح الكل</Button>
        </div>
        <div className="space-y-4 p-4">
          {records.length === 0 ? <p className="pt-12 text-center leading-8 text-muted-foreground">مازال ما درتش حتى تحليل. صوّر ورقة ونتلاقاو هنا مع العرض والنتيجة.</p> : records.map(item => <article key={item.id} className="history-card">
            <Button type="button" variant="ghost" onClick={() => { setCrop(item.crop); setPhoto(item.image); setAnalysis(item.analysis); setScreen('result'); }} className="h-auto w-full justify-start gap-3 whitespace-normal p-3 text-right hover:bg-accent">
              <img src={item.image} alt={`ورقة ${item.crop}`} className="h-18 w-18 shrink-0 rounded-xl object-cover" />
              <div className="min-w-0 flex-1"><div className="mb-1 flex flex-wrap items-center gap-1 text-xs"><span className="rounded-full bg-accent px-2 py-1 font-bold text-primary">{item.crop}</span><span className="rounded-full bg-primary px-2 py-1 font-bold text-primary-foreground">{item.analysis.status === 'healthy' ? 'صحيحة' : item.analysis.status === 'diseased' ? 'تحتاج عناية' : 'غير واضحة'}</span></div><h2 className="font-bold">{item.analysis.title}</h2><p className="line-clamp-2 text-xs leading-5 text-muted-foreground">{item.analysis.description}</p><p className="mt-1 text-[11px] text-muted-foreground">{new Date(item.date).toLocaleString('ar-DZ')}</p></div>
            </Button>
            <Button type="button" variant="ghost" onClick={() => saveRecords(records.filter(record => record.id !== item.id))} className="h-10 w-full rounded-none border-t text-destructive hover:bg-muted"><Trash2 size={15} /> حذف</Button>
          </article>)}
        </div>
      </div>}
    </main>
    {showIntro && <div className="intro-screen" role="status" aria-label="Fellah AI يفتح التطبيق">
      <picture className="intro-picture">
        <img src={introPortrait} alt="فلاح في مزرعته يفحص نباتاته" width={1088} height={1920} className="intro-photo" fetchPriority="high" />
      </picture>
      <div className="intro-shade" />
      <div className="intro-brand">
        <h1 dir="ltr">Fellah AI</h1>
        <p>فلاحك الذكي</p>
        <span className="intro-line" aria-hidden="true" />
      </div>
    </div>}
    </>
  );
}
