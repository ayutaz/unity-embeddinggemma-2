using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Hashes the same ordered raw coefficient bytes without copying weights or running a model.</summary>
    public static class ModelCoefficientHash
    {
        public static string Compute(Model model,out string backend,Func<Stream,string> accelerated=null)
        {
            if(model==null)throw new ArgumentNullException(nameof(model));
            var field=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
            if(field==null)throw new NotSupportedException("Pinned coefficient backing bytes are unavailable.");
            using var stream=new Segments(model.constants.Select(value=>(ArraySegment<byte>)field.GetValue(value)).ToArray());
            return PlayerFileHash.Compute(stream,out backend,accelerated);
        }
        sealed class Segments : Stream
        {
            readonly ArraySegment<byte>[] segments;
            readonly long[] starts;
            long position;int current;bool disposed;
            public Segments(ArraySegment<byte>[] segments)
            {
                this.segments=segments;starts=new long[segments.Length+1];
                for(var i=0;i<segments.Length;i++)starts[i+1]=checked(starts[i]+segments[i].Count);
            }
            void Check(){if(disposed)throw new ObjectDisposedException(nameof(Segments));}
            public override bool CanRead=>!disposed;
            public override bool CanSeek=>!disposed;
            public override bool CanWrite=>false;
            public override long Length{get{Check();return starts[starts.Length-1];}}
            public override long Position
            {
                get{Check();return position;}
                set{Check();if(value<0||value>Length)throw new ArgumentOutOfRangeException(nameof(value));position=value;current=0;while(current<segments.Length&&starts[current+1]<=position)current++;}
            }
            public override int Read(byte[] buffer,int offset,int count)
            {
                Check();if(buffer==null)throw new ArgumentNullException(nameof(buffer));
                if(offset<0||count<0||offset>buffer.Length||count>buffer.Length-offset)throw new ArgumentException("Read buffer bounds differ.");
                var total=0;
                while(count>0&&position<Length)
                {
                    while(current<segments.Length&&starts[current+1]<=position)current++;
                    var segment=segments[current];var at=(int)(position-starts[current]);var take=Math.Min(count,segment.Count-at);
                    Buffer.BlockCopy(segment.Array,segment.Offset+at,buffer,offset,take);offset+=take;count-=take;total+=take;position+=take;
                }
                return total;
            }
            public override long Seek(long offset,SeekOrigin origin)
            {
                Check();var start=origin==SeekOrigin.Begin?0:origin==SeekOrigin.Current?position:origin==SeekOrigin.End?Length:throw new ArgumentException("Seek origin differs.");
                Position=checked(start+offset);return position;
            }
            public override void Flush(){Check();}
            public override void SetLength(long value)=>throw new NotSupportedException();
            public override void Write(byte[] buffer,int offset,int count)=>throw new NotSupportedException();
            protected override void Dispose(bool disposing){disposed=true;base.Dispose(disposing);}
        }
    }
}
