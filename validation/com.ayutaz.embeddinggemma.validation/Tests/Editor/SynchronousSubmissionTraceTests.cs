using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using Newtonsoft.Json.Linq;
namespace EmbeddingGemma.Validation.Tests
{
    // Pure callback semantics; no Unity/GPU timing acceptance.
    public sealed class SynchronousSubmissionTraceTests
    {
        [Test] public void TypedNullExceptionNameDoesNotAbortSuccessfulSubmission()
        {
            Exception error=null;var row=new JObject();row["error_type"]=error?.GetType().FullName;
            Assert.That(SynchronousSubmissionTrace.HasError(row),Is.False);
            Assert.That(SynchronousSubmissionTrace.HasError(JObject.Parse(row.ToString())),Is.False);
        }
        [Test] public void ExplicitJsonNullDoesNotAbortSuccessfulSubmission()
        {
            Assert.That(SynchronousSubmissionTrace.HasError(new JObject { ["error_type"]=JValue.CreateNull() }),Is.False);
        }
        [Test] public void ActualExceptionNameAbortsSubmissionBeforeAndAfterJsonRoundTrip()
        {
            var row=new JObject { ["error_type"]=typeof(InvalidOperationException).FullName };
            Assert.That(SynchronousSubmissionTrace.HasError(row),Is.True);
            Assert.That(SynchronousSubmissionTrace.HasError(JObject.Parse(row.ToString())),Is.True);
        }
        [Test] public void MissingErrorFieldCannotMasqueradeAsSuccessfulSubmission()
        {
            Assert.Throws<ArgumentNullException>(()=>SynchronousSubmissionTrace.HasError(null));
            Assert.Throws<ArgumentException>(()=>SynchronousSubmissionTrace.HasError(new JObject()));
        }
        [Test] public void MalformedErrorFieldIsNotAcceptedAsSuccessOrAnOrdinaryStageFailure()
        {
            foreach(var value in new JToken[]{new JValue(false),new JValue(0),new JObject(),new JArray()})
                Assert.Throws<ArgumentException>(()=>SynchronousSubmissionTrace.HasError(new JObject { ["error_type"]=value }));
        }
        [Test] public void EmptyExceptionNameIsMalformedRatherThanASuccess()
        {
            foreach(var value in new[]{""," "})
                Assert.Throws<ArgumentException>(()=>SynchronousSubmissionTrace.HasError(new JObject { ["error_type"]=value }));
        }
        [Test] public void SynchronousResultAndScopeOrderArePreserved()
        {
            var result=new object();var events=new List<string>();
            var actual=SynchronousSubmissionTrace.Invoke(()=>{events.Add("invoke");return result;},()=>events.Add("begin"),error=>{Assert.That(error,Is.Null);events.Add("end");});
            Assert.That(actual,Is.SameAs(result));Assert.That(events,Is.EqualTo(new[]{"begin","invoke","end"}));
        }
        [Test] public void PendingTaskClosesSubmissionBeforeAsyncCompletionWithoutReplacingTheTask()
        {
            var pending=new TaskCompletionSource<object>();var ended=false;
            var actual=SynchronousSubmissionTrace.Invoke(()=>pending.Task,()=>{},error=>{Assert.That(error,Is.Null);ended=true;});
            Assert.That(actual,Is.SameAs(pending.Task));Assert.That(ended,Is.True);Assert.That(pending.Task.IsCompleted,Is.False);
            pending.SetResult(new object());Assert.That(pending.Task.IsCompleted,Is.True);
        }
        [Test] public void OriginalFailureIsPreservedEvenIfTheEndObserverAlsoFails()
        {
            var original=new InvalidOperationException("controlled original");var endings=0;
            foreach(var observerFails in new[]{false,true})
            {
                var actual=Assert.Throws<InvalidOperationException>(()=>SynchronousSubmissionTrace.Invoke(()=>throw original,()=>{},error=>{Assert.That(error,Is.SameAs(original));endings++;if(observerFails)throw new ArgumentException("controlled observer");}));
                Assert.That(actual,Is.SameAs(original));
            }
            Assert.That(endings,Is.EqualTo(2));
        }
        [Test] public void FailedBeginDoesNotInvokeTheStageOrEndAnUnstartedScope()
        {
            var error=new InvalidOperationException("controlled begin");var calls=0;var endings=0;
            Assert.That(Assert.Throws<InvalidOperationException>(()=>SynchronousSubmissionTrace.Invoke(()=>{calls++;return null;},()=>throw error,_=>endings++)),Is.SameAs(error));
            Assert.That(calls,Is.Zero);Assert.That(endings,Is.Zero);
        }
        [Test] public void FailedEndAfterSuccessfulStageIsNotSilentlyAccepted()
        {
            var error=new InvalidOperationException("controlled end");var calls=0;
            Assert.That(Assert.Throws<InvalidOperationException>(()=>SynchronousSubmissionTrace.Invoke(()=>{calls++;return null;},()=>{},_=>throw error)),Is.SameAs(error));
            Assert.That(calls,Is.EqualTo(1));
        }
        [Test] public void InvalidArgumentsAreRejectedBeforeAnyObserverOrStageRuns()
        {
            var calls=0;Func<object> invoke=()=>{calls++;return null;};Action begin=()=>calls++;Action<Exception> end=_=>calls++;
            Assert.Throws<ArgumentNullException>(()=>SynchronousSubmissionTrace.Invoke(null,begin,end));
            Assert.Throws<ArgumentNullException>(()=>SynchronousSubmissionTrace.Invoke(invoke,null,end));
            Assert.Throws<ArgumentNullException>(()=>SynchronousSubmissionTrace.Invoke(invoke,begin,null));Assert.That(calls,Is.Zero);
        }
    }
}
