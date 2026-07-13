import Stripe from "stripe";
import { supabaseAdmin } from "./_shared/supabaseAdmin";

export default async (req: Request) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });

  const stripeSecret = process.env.STRIPE_SECRET_KEY;
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!stripeSecret || !webhookSecret) {
    return new Response("Stripe is not configured.", { status: 500 });
  }
  const stripe = new Stripe(stripeSecret);

  const signature = req.headers.get("stripe-signature");
  if (!signature) return new Response("Missing signature", { status: 400 });

  // Signature verification needs the exact raw request body -- read it as
  // text, never JSON.parse it first.
  const rawBody = await req.text();

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, signature, webhookSecret);
  } catch (err) {
    return new Response(`Webhook signature verification failed: ${err instanceof Error ? err.message : String(err)}`, { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const userId = session.metadata?.user_id;
    const courseSlug = session.metadata?.course_slug;

    if (userId && courseSlug) {
      // Upsert on (user_id, course_slug) makes this safe against Stripe's
      // webhook retries -- a duplicate delivery just re-writes the same row.
      const { error } = await supabaseAdmin()
        .from("enrollments")
        .upsert(
          { user_id: userId, course_slug: courseSlug, status: "active", stripe_checkout_session_id: session.id },
          { onConflict: "user_id,course_slug" }
        );
      if (error) {
        console.error("Failed to record enrollment for", userId, courseSlug, error);
      }
    } else {
      console.error("checkout.session.completed missing user_id/course_slug metadata", session.id);
    }
  }

  return new Response("ok", { status: 200 });
};
