using System;
namespace EmbeddingGemma.Validation.Tests
{
    public static class SynchronousSubmissionTrace
    {
        /// <summary>Observe only invocation through its return, including a still-pending Task; never await here.</summary>
        public static object Invoke(Func<object> invoke, Action begin, Action<Exception> end)
        {
            if(invoke==null)throw new ArgumentNullException(nameof(invoke));
            if(begin==null)throw new ArgumentNullException(nameof(begin));
            if(end==null)throw new ArgumentNullException(nameof(end));
            begin();Exception original=null;
            try{return invoke();}
            catch(Exception error){original=error;throw;}
            finally
            {
                try{end(original);}
                catch when(original!=null){/* Keep the original invocation failure. */}
            }
        }
    }
}
