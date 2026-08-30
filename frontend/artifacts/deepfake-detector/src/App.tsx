import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import Home from '@/pages/Home';
import Console from '@/pages/Console';
import { TransitionProvider } from '@/context/TransitionContext';

const queryClient = new QueryClient();

// TransitionProvider must live inside WouterRouter so it can call useLocation()
function Router() {
  return (
    <TransitionProvider>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/console" component={Console} />
        <Route component={NotFound} />
      </Switch>
    </TransitionProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
