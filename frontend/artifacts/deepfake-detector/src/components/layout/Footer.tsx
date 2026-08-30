import { Github, FileText, Twitter, Mail, Shield } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-background py-12 px-6">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-cyan-500" />
          <span className="font-heading font-bold text-lg text-white">VERITAS</span>
        </div>

        <div className="flex flex-wrap justify-center gap-8 text-sm text-slate-400">
          <a href="#" className="hover:text-cyan-400 transition-colors flex items-center gap-2">
            <Github className="w-4 h-4" /> GitHub
          </a>
          <a href="#" className="hover:text-cyan-400 transition-colors flex items-center gap-2">
            <FileText className="w-4 h-4" /> Documentation
          </a>
          <a href="#" className="hover:text-cyan-400 transition-colors flex items-center gap-2">
            <FileText className="w-4 h-4" /> Research Paper
          </a>
          <a href="#" className="hover:text-cyan-400 transition-colors flex items-center gap-2">
            <Twitter className="w-4 h-4" /> X (Twitter)
          </a>
          <a href="#" className="hover:text-cyan-400 transition-colors flex items-center gap-2">
            <Mail className="w-4 h-4" /> Contact
          </a>
        </div>
      </div>
      <div className="max-w-6xl mx-auto mt-8 text-center text-xs text-slate-600">
        &copy; 2025 VERITAS. AI-Powered Deepfake Detection. For informational purposes only.
      </div>
    </footer>
  );
}
