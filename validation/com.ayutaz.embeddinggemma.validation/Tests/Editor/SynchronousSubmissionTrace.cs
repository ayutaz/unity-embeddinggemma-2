using System;
using Newtonsoft.Json.Linq;
namespace EmbeddingGemma.Validation.Tests
{
    public static class SynchronousSubmissionTrace
    {
        public static bool HasError(JObject row)
        {
            if(row==null)throw new ArgumentNullException(nameof(row));
            var token=row["error_type"];
            if(token==null)throw new ArgumentException("Submission error_type is missing.",nameof(row));
            if(token.Type==JTokenType.Null)return false;
            if(token.Type!=JTokenType.String)throw new ArgumentException("Submission error_type must be a string or null.",nameof(row));
            // Newtonsoft can retain JTokenType.String for an assigned null string.
            var name=token.Value<string>();
            if(name==null)return false;
            if(string.IsNullOrWhiteSpace(name))throw new ArgumentException("Submission error_type is empty.",nameof(row));
            return true;
        }

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
